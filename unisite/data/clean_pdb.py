import os
import shutil
import argparse
from tqdm import tqdm
from pymol import cmd
import pymol
pymol.finish_launching(['pymol', '-qc'])


def clean_protein(protein_path, output_dir):
    structure_name = os.path.basename(protein_path)
    cmd.reinitialize()
    cmd.load(protein_path, "protein_load")
    cmd.select("polymer_protein", "polymer.protein")
    protein_output_path = os.path.join(output_dir, structure_name)
    cmd.save(protein_output_path, "polymer_protein")
    return protein_output_path


def clean_all_pdbs(input_path, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    if os.path.isfile(input_path) and input_path.endswith('.pdb'):
        print(f"Cleaning PDB file: {input_path}")
        clean_protein(input_path, output_folder)
    elif os.path.isdir(input_path):
        pdb_files = [f for f in os.listdir(input_path) if f.endswith('.pdb')]
        print(f"Cleaning {len(pdb_files)} PDB files from {input_path} to {output_folder}...")
        for pdb in tqdm(pdb_files, desc="Cleaning PDB files"):
            pdb_path = os.path.join(input_path, pdb)
            clean_protein(pdb_path, output_folder)
    else:
        print("Error: Input must be a PDB file or a folder containing PDB files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='./', help="Single PDB file path or folder path containing PDB files")
    parser.add_argument("--output", default="./cleaned_pdbs", help="Folder to store cleaned PDB files")
    args = parser.parse_args()

    clean_all_pdbs(args.input, args.output)