"""
Functions to calculate mAP score.
Modified from mmdet.core.evaluation.mean_ap
"""
from multiprocessing import Pool

import numpy as np


def mask_overlaps(masks1, masks2):
    """
    Calculate the ious between each mask of masks1 and masks2.
    
    Args:
        masks1: shape (n, l)
        masks2: shape (k, l)
    
    Returns:
        ious: shape (n, k)
    """
    masks1 = masks1.astype(bool)
    masks2 = masks2.astype(bool)
    intersection = masks1[:, None, :] & masks2[None, :, :]
    union = masks1[:, None, :] | masks2[None, :, :]
    ious = intersection.sum(axis=-1) / (union.sum(axis=-1) + 1e-6)
    return ious


def average_precision(recalls, precisions, mode="area"):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == "area":
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == "11points":
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError('Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_default(
    pred_masks, pred_scores, gt_masks, iou_thr=0.5
):
    """Check if detected bboxes are true positive or false positive.

    Args:
        pred_masks (ndarray): predicted masks of this sequence, of shape (m, l).
        pred_scores (ndarray): predicted scores of masks, of shape (m,).
        gt_masks (ndarray): GT pocket masks of this sequence, of shape (n, l).
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (m,).
    """

    num_dets = pred_masks.shape[0]
    num_gts = gt_masks.shape[0]

    tp = np.zeros((num_dets,), dtype=np.float32)
    fp = np.zeros((num_dets,), dtype=np.float32)

    # if there is no gt masks in this image, then all pred_masks are false positives
    if gt_masks.shape[0] == 0:
        fp[...] = 1
        return tp, fp
    if num_dets == 0:
        return tp, fp

    ious = mask_overlaps(pred_masks, gt_masks)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-pred_scores)

    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in sort_inds:
        if ious_max[i] >= iou_thr:
            matched_gt = ious_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    return tp, fp


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): class id

    Returns:
        tuple[list[np.ndarray]]: cls_dets, cls_scores, cls_gts
    """
    cls_dets = []
    cls_scores = []
    cls_gts = []
    for i in range(len(det_results)):
        gt_inds = annotations[i]["labels"] == class_id
        cls_gt = annotations[i]["pocket_masks"][gt_inds, :]
        if len(det_results[i]["labels"]) == 0:
            cls_dets.append(np.zeros((0, cls_gt.shape[1]), dtype=np.float32))
            cls_scores.append(np.zeros((0,), dtype=np.float32))
            cls_gts.append(cls_gt)
            continue
        cls_inds = det_results[i]["labels"] == class_id
        cls_det = det_results[i]["pocket_masks"][cls_inds, :]
        if "res_mask" in annotations[i]:
            res_mask = annotations[i]["res_mask"]
            cls_det = cls_det[:, res_mask]
            cls_gt = cls_gt[:, res_mask]

        cls_dets.append(cls_det)
        cls_scores.append(det_results[i]["scores"][cls_inds])
        cls_gts.append(cls_gt)

    return cls_dets, cls_scores, cls_gts

def eval_map(
    det_results,
    annotations,
    num_classes,
    iou_thr=0.7,
    nproc=4,
    fixed_recall=[0.7],
):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
                - "bboxes": numpy array of shape (n, 4)
                - "labels": numpy array of shape (n, )
                - "bboxes_ignore" (optional): numpy array of shape (k, 4)
                - "labels_ignore" (optional): numpy array of shape (k, )
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    if len(det_results) == 0:
        print("No detection results.")
        return np.nan, [], []
    # print(f"There are {num_classes} classes.")

    pool = Pool(nproc)
    eval_results = []
    ret_thres = []
    for i in range(num_classes):
        # get gt and det masks of this class
        cls_dets, cls_scores, cls_gts = get_cls_results(det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        tpfp_func = tpfp_default
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_func,
            zip(
                cls_dets,
                cls_scores,
                cls_gts,
                [iou_thr for _ in range(num_imgs)],
            ),
        )
        tp, fp = tuple(zip(*tpfp))

        num_gts = sum([gt.shape[0] for gt in cls_gts])

        # sort all det by score, also sort tp and fp
        cls_scores = np.hstack(cls_scores)
        num_dets = len(cls_scores)
        sort_inds = np.argsort(-cls_scores)
        probs = cls_scores[sort_inds]
        tp = np.hstack(tp)[sort_inds]
        fp = np.hstack(fp)[sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        # print("num_gts", num_gts)
        precisions = tp / np.maximum((tp + fp), eps)
        fpr = fp / np.maximum((tp + fp), eps)
        fps = fp / num_imgs
        # assert recalls.shape[0] == 1, "Multiple recall rows detected!"
        rec = recalls
        ret_thre = np.interp(fixed_recall, rec, probs) if len(rec) else 0
        ret_thres.append(ret_thre)
        # calculate AP
        mode = "area" # if dataset != "voc07" else "11points"
        ap = average_precision(recalls, precisions, mode)
        eval_results.append(
            {
                "num_gts": num_gts,
                "num_dets": num_dets,
                "recall": recalls,
                "precision": precisions,
                "score": probs,
                "fpr": fpr,
                "fps": fps,
                "ap": ap,
            }
        )

    aps = []
    for cls_result in eval_results:
        if cls_result["num_gts"] > 0:
            aps.append(cls_result["ap"])
    mean_ap = np.array(aps).mean().item() if aps else 0.0
    # assert len(ret_thres), "num_classes == 0!!"
    return mean_ap, eval_results, ret_thres
