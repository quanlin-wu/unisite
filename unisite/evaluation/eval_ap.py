import os
import torch
import sys
import pickle
import argparse
import mean_ap
from unisite.data.dataset import Pockets


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the inference results')
    parser.add_argument('-e', '--eval_dir', type=str, required=True, help='Directory containing results to evaluate')
    parser.add_argument('-t', '--target_dir', type=str, help='Directory containing target data', required=True)
    return parser.parse_args()


def load_target(pkl_path):
    data = pickle.load(open(pkl_path, 'rb'))
    if "target" in data.keys():
        return data["target"]


if __name__ == "__main__":
    args = parse_args()
    target_dir = args.target_dir
    eval_dir = args.eval_dir
    iou_thresholds = [0.3, 0.5] 

    infer_results = os.listdir(eval_dir)
    infer_results = [os.path.join(eval_dir, x) for x in infer_results if x.endswith('.pkl')]
    predictions = []
    targets = []
    for result in infer_results:
        with open(result, 'rb') as f:
            data = pickle.load(f)
        if 'target' in data:
            targets.append(data.pop('target'))
        else:
            pkl_path = os.path.join(target_dir, os.path.basename(result))
            target = load_target(pkl_path)
            if len(data["pocket_masks"]) > 0 and target["pocket_masks"].shape[1] != data["pocket_masks"].shape[1]:
                print(f"Warning: target and prediction length mismatch for {result}")
                continue
            targets.append(target)
        predictions.append(data)
    print(f"Number of predictions: {len(predictions)}")

    ckpt_metrics = {}
    for iou_thr in iou_thresholds:
        map, eval_results, ret_thres = mean_ap.eval_map(predictions, targets, 1, iou_thr=iou_thr)
        ckpt_metrics.update({
            f'AP_class_0@IOU_{iou_thr}': eval_results[0]['ap'],
        })
        print(f'AP_class_0@IOU_{iou_thr}: {eval_results[0]["ap"]}')
