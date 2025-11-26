In this document, we provide a detailed explanation and usage instructions for the evaluation scripts included in the UniSite project. These scripts are designed to assess the performance of ligand binding site prediction models using various metrics.

To run the evaluation scripts, ensure that you have the necessary dependencies installed and that your predicted binding site data is organized according to the expected format. The dependencies can be installed according to the instructions provided in the main README.md file. And the datasets can be downloaded at [here](https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1/tree/main), with detailed illustration in [DATASETS.md](./DATASETS.md).

## IoU-based AP Evaluation

In this work, we introduce **Average Precision (AP) based on Intersection over Union (IoU)** as a more accurate evaluation metric for ligand binding site prediction.
We calculate AP as follows: First, we sort all predictions by confidence scores. Then, we match each ground truth site to the predicted site with the highest score and residue-level IoU above a predetermined threshold, enforcing a one-to-one assignment constraint. Finally, we compute AP as the area under the interpolated precision-recall curve following COCO evaluation protocols, which is widely used in object detection.
The pseudo-code for AP calculation is provided in our [paper](https://arxiv.org/pdf/2506.03237) (Appendix K). The AP metric offers two significant
advantages: (1) the residue-level IoU enables accurate shape and size comparison between binding
sites; (2) the one-to-one matching scheme inherently prevents double-counting of predictions.

To evaluate your predictions with IoU-based AP use the following command:

```bash
conda activate unisite
python unisite/evaluation/eval_ap.py -e {EVALUATION_DIR} -t {GROUND_TRUTH_DIR}
```

- `{EVALUATION_DIR}`: Directory containing the predicted binding site files. Each file should be named the same as the corresponding target file, and contains key-value pairs as follows (assumpting N predicted sites):
  
  - "labels": numpy.array of shape (N,), class labels for AP calculation (all set to 0 currently).
  - "scores": numpy.array of shape (N,), confidence scores for each predicted binding site.
  - "pocket_masks": numpy.array of shape (N, L), binary masks indicating the residues belonging to each predicted binding site, where L is the length of the protein sequence.
- `{GROUND_TRUTH_DIR}`: Directory containing the ground truth binding site files. these files should be organized in the same format as in the **pkl_files** folder of UniSite-DS dataset.

We provide some results of UniSite in [here](https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1/tree/main) as a reference. You can use them to reproduce the AP results of UniSite-3D reported in our paper.

## DCC and DCA Evaluation

**DCC** (Distance between the predicted binding site center and the true binding site center) and **DCA** (Shortest distance between the predicted binding site center and any heavy atom of the ligand) are the two most widely-used metrics for binding site detection. A binding site prediction is considered successful when its DCC or DCA value is below a predetermined threshold. Previous works quantify prediction performance via the Success Rate, defined as the ratio of successful predictions to the total number of ground truth sites:

$$
\text{Success Rate (DCC or DCA)} = \frac{|\lbrace\text{Predicted sites | DCC or DCA < threshold}\rbrace|}{|\lbrace\text{Ground truth sites}\rbrace|}
$$

Since the DCC and DCA metrics do not consider prediction confidence scores or ranks,  a common approach is to calculate the DCC or DCA for either the top-n or top-(n+2) predicted binding sites, where n is the number of ground truth sites. In this work, we follow the prior works and report the Success Rate based on the top-n and top-(n+2) predictions.

To evaluate your predictions with DCC and DCA metrics, use the following command:

```bash
conda activate unisite
python unisite/evaluation/eval_dcc_dca.py -e {EVALUATION_DIR} -t {TARGET_CSV}
```

- `{EVALUATION_DIR}`: Directory containing the predicted binding site files. Each file should be named the same as the corresponding target (in PKL format), and contains key-value pairs as follows (assumpting N predicted sites):
  
  - "scores": numpy.array of shape (N,), confidence scores for each predicted binding site.
  - "centers": numpy.array of shape (N, 3), the 3D coordinates of the predicted centers of each prediction.
- `{TARGET_CSV}`: A CSV file containing the ground truth binding site information for all proteins being evaluated. The CSV file can be organized in the same format as the **holo4k-sc.csv** or **coach420.csv** files in the benchmark datasets. Specifically, it should contain the following columns:
  
  - name: f"{PDB ID}{Chain ID}" of the structure.
  - pdb_file_path: the file path of the protein structure (relative path from the dataset root, e.g. holo4k-sc/1a2bA/1a2bA_protein.pdb).

We provide some results of UniSite in [here](https://huggingface.co/datasets/quanlin-wu/unisite-ds_v1/tree/main) as a reference. You can use them to reproduce the DCC & DCA results of UniSite-3D on HOLO4K-sc reported in our paper.

