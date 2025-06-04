"""Pytorch script for training protein pocket detector.
"""
import os
import torch
import GPUtil
import time
import tree
import numpy as np
import pandas as pd
import copy
import hydra
import logging
import copy
import pickle

from datetime import datetime, timedelta
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch.utils
import torch.distributed as dist
import esm

from hydra.core.hydra_config import HydraConfig

from unisite.data.dataset import InferDataset, get_esm_embedding, length_collate, map_to
from unisite.model import detector
from unisite.model.utils import get_df_from_dict


def get_ddp_info():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    node_id = rank // world_size
    return {"node_id": node_id, "local_rank": local_rank, "rank": rank, "world_size": world_size}


class Experiment:

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = [str(x) for x in GPUtil.getAvailable(
                order='memory', limit = 16)]

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        self._model_conf = conf.model
        self._data_conf = conf.data

        if self._exp_conf.use_gpu and self._exp_conf.num_gpus > 1:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend='nccl', timeout=timedelta(hours=1))
            self.ddp_info = get_ddp_info()
            if self.ddp_info['rank'] not in [0,-1]:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
        else:
            self.ddp_info = {'node_id': 0, 'local_rank': 0, 'rank': 0, 'world_size': 1}

        # Initialize experiment objects
        self._model = detector.DetectorNetwork(self._model_conf)
        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f'Number of model parameters {num_parameters}')

        checkpoint = torch.load(conf.ckpt_path, weights_only=False)
        self._log.info(f'Load checkpoint from: {conf.ckpt_path}')
        ckpt_model = checkpoint['model']
        ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_model.items()}
        self._model.load_state_dict(ckpt_model, strict=True)
        if self._exp_conf.use_gpu and self._exp_conf.num_gpus > 1:
            device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device
        self._model = self._model.to(device)

        assert self._exp_conf.output_dir is not None
        self._exp_conf.eval_dir = self._exp_conf.output_dir
        os.makedirs(self._exp_conf.eval_dir, exist_ok=True)
        self._log.info(f"Results saved to: {self._exp_conf.eval_dir}")

        if dist.get_rank() == 0:
            config_path = os.path.join(self._exp_conf.output_dir, 'inference_conf.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._conf, f=f)
            self._log.info(f'Saving inference config to {config_path}')
    
    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf
    
    def create_dataset(self):
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        esm_path = os.path.join(repo_path, 'model_weights/esm2/esm2_t33_650M_UR50D.pt')
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet(esm_path)
        esm_model = esm_model.to(self._device)
        esm_model.eval()
        batch_converter = alphabet.get_batch_converter()

        if self._data_conf.test_data.endswith('.csv'):
            df = pd.read_csv(self._data_conf.test_data)
        elif self._data_conf.test_data.endswith('.pdb'):
            df = pd.DataFrame({'name': [os.path.basename(self._data_conf.test_data)[:-4]],
                            'pdb_file_path': [self._data_conf.test_data]})
            assert self._model_conf.model_type != "seq_only"
        elif self._data_conf.test_data.endswith('.fasta'):
            df_rows = []
            with open(self._data_conf.test_data, 'r') as f:
                name = None
                seq_slices = []
                for line in f:
                    if line.startswith('>'):
                        if len(seq_slices) > 0:
                            seq = ''.join(seq_slices)
                            df_rows.append({'name': name, 'sequence': seq})
                            seq_slices = []
                        name = line[1:].strip()
                    else:
                        seq_slices.append(line.strip())
                if len(seq_slices) > 0:
                    seq = ''.join(seq_slices)
                    df_rows.append({'name': name, 'sequence': seq})
            df = pd.DataFrame(df_rows)
            assert self._model_conf.model_type == "seq_only"
        elif os.path.isdir(self._data_conf.test_data):
            pdb_files = [f for f in os.listdir(self._data_conf.test_data) if f.endswith('.pdb')]
            df_rows = [
                {'name': f[:-4], 'pdb_file_path': os.path.join(self._data_conf.test_data, f)}
                for f in pdb_files
            ]
            df = pd.DataFrame(df_rows)
            assert self._model_conf.model_type != "seq_only"
        else:
            raise ValueError(f"Unsupported test data format: {self._data_conf.test_data}")

        # Dataset
        valid_dataset = InferDataset(
            dataset_df=df,
            esm_func=lambda x: get_esm_embedding(x, esm_model, batch_converter, self._device),
            seq_only=(self._model_conf.model_type == "seq_only"),
        )
        # Sampler
        if self._exp_conf.use_gpu and self._exp_conf.num_gpus > 1:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
            )
            batch_size = self._exp_conf.eval_batch_size // num_replicas
        else:
            valid_sampler = None
            batch_size = self._exp_conf.eval_batch_size
        # Loaders
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=length_collate,
        )
        print(f"Device {self._device} has {len(valid_loader)} batches")
        return valid_loader, valid_sampler


    def start_inference(self):
        valid_loader, valid_sampler = self.create_dataset()

        if self.ddp_info['rank'] in [0, -1]:
            self._log.info(f'Start inference')
        # Run inference parallelly
        if self._exp_conf.use_gpu and self._exp_conf.num_gpus > 1:
            dist.barrier()
        start_time = time.time()
        eval_dir = self._exp_conf.eval_dir
        os.makedirs(eval_dir, exist_ok=True)
        self.inference_fn(valid_loader, eval_dir, self._device)
        if self._exp_conf.use_gpu and self._exp_conf.num_gpus > 1:
            dist.barrier()
        infer_time = time.time() - start_time
        if self.ddp_info['rank'] in [0, -1]:
            self._log.info(f'Finished inference in {infer_time:.2f}s')

    def inference_fn(
            self,
            valid_loader,
            eval_dir,
            device,
        ):
        self.model.eval()
        for valid_feats in valid_loader:
            valid_feats = map_to(valid_feats, device)
            with torch.no_grad():
                model_out = self.model(valid_feats)
            predictions = self.model.post_process(model_out, valid_feats['length'], return_numpy=True)
            for i in range(len(predictions)):
                pred = predictions[i]
                name = valid_feats['name'][i]
                if 'pdb_file_path' in valid_feats:
                    pred['pdb_file_path'] = valid_feats['pdb_file_path'][i]
                if 'map_dict' in valid_feats:
                    pred['map_dict'] = valid_feats['map_dict'][i]
                if 'target' in valid_feats:
                    pred['target'] = valid_feats['target'][i].numpy()
                pickle.dump(pred, open(os.path.join(eval_dir, f'{name}.pkl'), 'wb'))
                df = get_df_from_dict(pred)
                df.to_csv(os.path.join(eval_dir, f'{name}.csv'), index=False)          
        self.model.train()
        return eval_dir


@hydra.main(version_base=None, config_path="../config", config_name="inference")
def run(conf: DictConfig) -> None:
    exp = Experiment(conf=conf)
    exp.start_inference()


if __name__ == '__main__':
    run()