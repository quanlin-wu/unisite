import numpy as np


def calc_dca_dcc_metrics(pred_centers_list, pred_scores_list, ligs_list, top_n_plus=0):
    dca_results = []
    dcc_results = []
    assert len(pred_centers_list) == len(pred_scores_list)
    assert len(pred_centers_list) == len(ligs_list)
    for i in range(len(pred_centers_list)):
        pred_centers = pred_centers_list[i]
        pred_scores = pred_scores_list[i]
        ligs = ligs_list[i].values()
        num_preds = len(pred_centers)
        num_ligs = len(ligs)
        if num_ligs == 0:
            dca_results.append([])
            dcc_results.append([])
            # print(f"Warning: No ligands found in the input for index {i}. Returning empty results.")
            continue
        if num_preds == 0:
            dca_results.append(np.full(num_ligs, np.inf))
            dcc_results.append(np.full(num_ligs, np.inf))
            continue

        # top_{num_ligs + top_n_plus}
        sorted_ids = np.argsort(pred_scores)[::-1]
        sorted_ids = sorted_ids[: num_ligs + top_n_plus]
        pred_centers = pred_centers[sorted_ids]
        num_preds = len(pred_centers)

        dcc_matrix = np.zeros((num_ligs, num_preds))
        dca_matrix = np.zeros((num_ligs, num_preds))
        for i, lig in enumerate(ligs):
            lig_coords = np.array([atom["coordinates"] for atom in lig])
            lig_center = np.mean(lig_coords, axis=0)
            dcc_matrix[i] = np.linalg.norm(pred_centers - lig_center, axis=1)
            distances = np.linalg.norm(pred_centers[:, None, :] - lig_coords[None], axis=-1)
            dca_matrix[i] = np.min(distances, axis=1)
        dcc_results.append(np.min(dcc_matrix, axis=1))
        dca_results.append(np.min(dca_matrix, axis=1))
    return dcc_results, dca_results

