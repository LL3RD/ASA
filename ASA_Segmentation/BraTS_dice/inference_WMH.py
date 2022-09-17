import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import dc, hd95
from multiprocessing.pool import Pool
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd


def compute_BraTS_dice(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    :param ref:
    :param gt:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 1
        else:
            return 0
    else:
        return dc(pred, ref)


def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1))


def evaluate_BraTS_case(arr: np.ndarray, arr_gt: np.ndarray):
    """
    attempting to reimplement the brats evaluation scheme
    assumes edema=1, non_enh=2, enh=3
    :param arr:
    :param arr_gt:
    :return:
    """
    # whole tumor
    mask_gt = (arr_gt == 1).astype(int)
    mask_pred = (arr == 1).astype(int)
    dc_whole = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return dc_whole, hd95_whole


def load_evaluate(filename_gt: str, filename_pred: str):
    arr_pred = sitk.GetArrayFromImage(sitk.ReadImage(filename_pred))
    arr_gt = sitk.GetArrayFromImage(sitk.ReadImage(filename_gt))
    return evaluate_BraTS_case(arr_pred, arr_gt)


def evaluate_BraTS_folder(folder_pred, folder_gt, num_processes: int = 24, strict=False):
    nii_pred = subfiles(folder_pred, suffix='.nii.gz', join=False)
    if len(nii_pred) == 0:
        return
    nii_gt = subfiles(folder_gt, suffix='.nii.gz', join=False)
    assert all([i in nii_gt for i in nii_pred]), 'not all predicted niftis have a reference file!'
    if strict:
        assert all([i in nii_pred for i in nii_gt]), 'not all gt niftis have a predicted file!'
    p = Pool(num_processes)
    nii_pred_fullpath = [join(folder_pred, i) for i in nii_pred]
    nii_gt_fullpath = [join(folder_gt, i) for i in nii_pred]
    results = p.starmap(load_evaluate, zip(nii_gt_fullpath, nii_pred_fullpath))
    # now write to output file
    with open(join(folder_pred, 'results.csv'), 'w') as f:
        f.write("name,DC_WMH,HD95_WMH\n")
        for fname, r in zip(nii_pred, results):
            f.write(fname)
            f.write(",%0.4f,%3.3f\n" % r)


def main():
    folder_pred = '/data2/huangjunjia/nnFormer/nnFormer_trained_models/nnFormer/3d_fullres/Task555_WMH/Paper/infers/TransBTS/'
    evaluate_BraTS_folder(folder_pred,
                          '/data2/huangjunjia/nnFormer/nnFormer_raw/nnFormer_raw_data/Task555_WMH/labelsTs/')

    result = pd.read_csv(join(folder_pred, 'results.csv'))[
        ['DC_WMH', 'HD95_WMH']]
    print(result.mean(axis=0))


if __name__ == '__main__':
    main()
