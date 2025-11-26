In this document, we provide the illustration for our newly curated UniSite-DS dataset, and the two benchmark datasets, HOLO4K-sc and COACH. All datasets can be downloaded at [here](https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1/tree/main).

## UniSite-DS

UniSite-DS is **the first UniProt (Unique Protein)-centric ligand binding site dataset**, which systematically integrates all ligand binding sites associated with given unique protein across multiple PDB structures. UniSite-DS contains 11,510 unique proteins and 3,670 multi-site unique proteins, which is 4.81 times more multi-site data and 2.08 times more overall data compared to the previously most widely used datasets. The details of dataset curation and statistics are illustrated in our [paper](https://arxiv.org/pdf/2506.03237).

The structure of UniSite-DS is as follows:

```
unisite-ds/
├── metadata.csv  # metadata file
├── A0A0A0BZU2    # a unique protein subfolder
├── A0A0A0MTJ0
├── A0A0A0Q7Z8/
|   ├── A0A0A0Q7Z8_info.csv # information of all binding sites of this unique protein
|   ├── A0A0A0Q7Z8.pdb
|   ├── A0A0A0Q7Z8.mapping  # mapping file between PDB sequence and UniProt sequence 
│   ├── site1/
|   |   ├── A0A0A0Q7Z8-6lc8A.mapping
|   |   ├── binding_site.pdb
|   |   ├── protein.pdb
|   |   ├── ligand.pdb
|   |   ├── ligand.sdf
|   |   └── complex.pdb
│   └── site2/
|       ├── A0A0A0Q7Z8-6lc9B.mapping
|       ├── ...
├── A0A0A0RCB5
├── ...
└── pkl_files/   # preprocessed data files for training and evaluation
    ├── A0A0A0BZU2.pkl
    ├── A0A0A0MTJ0.pkl
    ├── A0A0A0Q7Z8.pkl
    ├── A0A0A0RCB5.pkl
    ├── ...
    ├── train_0.9.csv   # split files  
    ├── test_0.9.csv
    └── ...
```

**metadata.csv**
contains the overall information of all unique proteins in UniSite-DS. The columns are as follows:

- unp_id: UniProt ID of the unique protein.
- seq: Amino acid sequence of the unique protein.
- seq_len: Length of the amino acid sequence.
- number_sites: Number of ligand binding sites associated with this unique protein.
- subdir: Subdirectory name of this unique protein (same as unp_id).

---
For each unique protein subfolder (e.g., P01116):

```
P01116/
├── P01116_info.csv
├── P01116.pdb
├── P01116.mapping
├── site1/
│   ├── P01116-8onvA.mapping
│   ├── binding_site.pdb
│   ├── protein.pdb
│   ├── ligand.pdb
│   ├── ligand.sdf
│   └── complex.pdb
├── site2/
│   ├── P01116-6bofA.mapping
│   ├── ...
└── ...
```

- **{unp_id}_info.csv**: Information of all binding sites of this unique protein. Each row corresponds to one binding site. The columns are as follows:
  
  - site: Binding site ID (e.g., site1, site2, ...).
  - pdb_id: PDB ID of the structure containing this binding site.
  - chain_id: Chain ID of the protein chain containing this binding site.
  - chain_id_ligand: Chain ID of the ligand chain.
  - ligand_name: Name of the ligand.
  - ligand_id: the Residue ID of the ligand in the PDB structure.
  - site_position_uniprot: Positions of the binding site residues in the UniProt sequence (1-based indexing).
  - site_position_pdb: Residue IDs of the binding site residues in the PDB structure.
- **{unp_id}.pdb**: The representative PDB structure of this unique protein (i.e., the  PDB structure with the highest sequence identity to the UniProt sequence).
- **{unp_id}.mapping**: The mapping file between the PDB sequence and the UniProt sequence of the representative structure. Each line contains two integers separated by a comma, representing the Residue ID in the PDB structure and the corresponding position in the UniProt sequence, respectively.
- **site{n}/**: Subfolder of the n-th binding site of this unique protein.
  
  - **binding_site.pdb**: PDB file of the binding site (only contains the binding site residues).
  - **protein.pdb**: PDB file of the pure binded protein chain.
  - **ligand.pdb**: PDB file of the ligand.
  - **ligand.sdf**: SDF file of the ligand.
  - **complex.pdb**: PDB file of the protein-ligand complex.
  - **{unp_id}-{pdb_id}{chain_id}.mapping**: The mapping file between the PDB sequence and the UniProt sequence for this specific structure.

---
**pkl_files/**: Preprocessed data files for training and evaluation.

- Each PKL file corresponds to one unique protein and contains the following fields:

  - "label": the UniProt ID of the unique protein.
  - "sequence": the amino acid sequence of the unique protein.
  - "pdb_file_path": the file path of the representative PDB structure (currently the relative path from the dataset root).
  - "map_dict": the mapping dictionary between the Residue ID in the PDB structure and the position in the UniProt sequence (1-based indexing).
  - "target": the target binding sites for training and evaluation, containing:
    - "labels": class labels for AP calculation (all set to 0 currently).
    - "pocket_masks": binary masks for each binding site. 1 indicates the residue is part of the binding site, and 0 otherwise. Shape (num_sites, seq_len).
    - "res_mask": binary mask indicating valid residues (1 for valid residues, 0 for padding). Shape (seq_len).

- The split files (*.csv) contain the list of unique protein IDs for each data split. **train/test_{sim_threshold}.csv** contain the training and test splits with different sequence identity thresholds. **train/test_benchmark.csv** contain the training and test splits for benchmark evaluation.

## Benchmark Datasets

HOLO4K and COACH420 are two benchmark datasets utilized for protein–ligand binding site detection (Source: https://github.com/rdk/p2rank-datasets). Follow the prior works, we employ the *mlig* subsets of these two dataset, which contain explicitly specified relevant ligands. We further select the single-chain structures, denoting the test datasets as HOLO4K-sc and COACH420 (all structures in COACH420 are originally single-chain).

The structure of benchmark datasets is as follows:

```
benchmark_datasets/
├── holo4k-sc/
│   ├── holo4k-sc.csv 
│   ├── holo4k-sc_seq.csv
│   ├── target_pkl/
│   │   ├── 1a2bA.pkl
│   │   ├── 1a5zA.pkl
│   │   ├── ...
│   ├──1a2bA/
│   │   ├── 1a2bA_protein.pdb
│   │   ├── 1a2bA_ligand1.pdb
│   │   ├── 1a2bA_pocket1.txt
│   │   ├── ...
│   ├── ...
└── coach420/
    ├── coach420.csv
    ├── coach420_seq.csv
    ├── target_pkl/
    └── ...
```

Takes HOLO4K-sc as an example,

- **holo4k-sc.csv**, contains columns as follows:

  - name: f"{PDB ID}{Chain ID}" of the structure.
  - pdb_file_path: the file path of the protein structure (relative path from the dataset root, e.g. holo4k-sc/1a2bA/1a2bA_protein.pdb).

- **holo4k-sc_seq.csv**, contains columns as follows:

  - name: f"{PDB ID}{Chain ID}" of the structure.
  - sequence: the amino acid sequence of the protein chain.

- **target_pkl/**: Preprocessed data files for evaluation. Each PKL file corresponds to one structure and contains the following fields:

  - "target": the target binding sites for training and evaluation, the same as UniSite-DS.
  - "centers": the geometric centers of the ligands in the binding sites. Shape: (num_sites, 3).

- Each structure subfolder (e.g., 1a2bA/) contains:

  - **{name}_protein.pdb**: PDB file of the pure protein chain.
  - **{name}_ligand{n}.pdb**: PDB file of the n-th ligand.
  - **{name}_pocket{n}.txt**: A text file containing the Residue IDs of the binding site residues for the n-th ligand, one per line.

