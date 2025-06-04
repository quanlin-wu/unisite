import pandas as pd
import os
import pickle
import numpy as np
import argparse
from rdkit import Chem
from dcc_dca import calc_dca_dcc_metrics


def load_ligand(ligand_file):
    mol = Chem.MolFromPDBFile(ligand_file, removeHs=True, sanitize=False)
    ligand = []
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atom_name = atom.GetSymbol()
        residue_id = atom.GetPDBResidueInfo().GetResidueNumber()
        ligand.append({"res_id": residue_id, "atom_name": atom_name, "coordinates": [pos.x, pos.y, pos.z]})
    return ligand


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the inference results')
    parser.add_argument('-e', '--eval_dir', type=str, required=True, help='Directory containing results to evaluate')
    parser.add_argument('-t', '--target_csv', type=str, help='CSV file directing to the target data', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_dir = args.eval_dir
    target_csv = args.target_csv

    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"Evaluation directory {eval_dir} does not exist.")
    if not os.path.exists(target_csv):
        raise FileNotFoundError(f"Target CSV file {target_csv} does not exist.")

    df = pd.read_csv(target_csv)
    ligands_list = []
    pred_centers_list = []
    pred_scores_list = []
    for _, row in df.iterrows():
        pdb_file_path = row["pdb_file_path"]
        ligand_files = os.listdir(os.path.dirname(pdb_file_path))
        ligand_files = [os.path.join(os.path.dirname(pdb_file_path), f) for f in ligand_files if "ligand" in f]
        ligand_files = sorted(ligand_files)
        ligands_list.append({f"ligand_{i+1}": load_ligand(ligand_file) for i, ligand_file in enumerate(ligand_files)})
        result = pickle.load(open(os.path.join(eval_dir, row["name"] + ".pkl"), "rb"))
        pred_scores_list.append(result["scores"])
        pred_centers_list.append(result.get("hull_centers", result.get("centers", [])))
    print(f"Number of predictions: {len(df)}")
        
    # top-n
    dcc_results, dca_results = calc_dca_dcc_metrics(pred_centers_list, pred_scores_list, ligands_list, top_n_plus=0)
    dcc_success_rate_topn = np.mean(np.concatenate(dcc_results) < 4)
    dca_success_rate_topn = np.mean(np.concatenate(dca_results) < 4)
    # top-(n+2)
    dcc_results, dca_results = calc_dca_dcc_metrics(pred_centers_list, pred_scores_list, ligands_list, top_n_plus=2)
    dcc_success_rate_topn_2 = np.mean(np.concatenate(dcc_results) < 4)
    dca_success_rate_topn_2 = np.mean(np.concatenate(dca_results) < 4)

    print(f"                 |  top-n  |  top-(n+2)")
    print(f"DCC success rate |  {dcc_success_rate_topn:.4f} |  {dcc_success_rate_topn_2:.4f}")
    print(f"DCA success rate |  {dca_success_rate_topn:.4f} |  {dca_success_rate_topn_2:.4f}")
