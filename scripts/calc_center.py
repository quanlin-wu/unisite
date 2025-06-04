import pickle
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import warnings

from concurrent.futures import ProcessPoolExecutor
from unisite.model.utils import calc_center
from unisite.data import load_pdb

warnings.filterwarnings("ignore")


def calc_center_func(args):
    try:
        pkl_file, csv_file = args
        result = pickle.load(open(pkl_file, "rb"))
        protein = load_pdb(result["pdb_file_path"])
        pred_masks = result["pocket_masks"]
        assert pred_masks.shape[1] == protein.num_residue
        ret_centers = calc_center(protein, pred_masks, methods=["hull"])
        result["centers"] = ret_centers["hull_centers"]
        pickle.dump(result, open(pkl_file, "wb"))
        keep = np.ones_like(result["scores"], dtype=bool)
        # Filter out centers with low scores or small pocket masks as in 'model.utils.get_df_from_dict'
        for i in range(len(result["centers"])):
            if result["scores"][i] <= 0 or sum(result["pocket_masks"][i]) <= 3:
                keep[i] = False
        df = pd.read_csv(csv_file)
        df["center"] = result["centers"][keep].tolist()
        df.to_csv(csv_file, index=False)
        return 1 # success
    except Exception as e:
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate centers of predicted pockets')
    parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directory containing predictions')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    results = os.listdir(input_dir)
    results_pkl = [os.path.join(input_dir, x) for x in results if x.endswith('.pkl')]
    input_list = [(pkl_file, pkl_file.replace(".pkl", ".csv")) for pkl_file in results_pkl]
    print(f"Total {len(input_list)} files to process.")
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(tqdm(executor.map(calc_center_func, input_list), total=len(input_list)))
    print(f"Processed {sum(results)} files successfully.")
    for i, result in enumerate(results):
        if result == 0:
            print(f"Failed to process file {input_list[i][1]}.")
