from copy import deepcopy
from multiprocessing.pool import Pool

import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *

import scipy.stats as ss

from nnformer.dataset_conversion.Task032_BraTS_2018 import convert_labels_back_to_BraTS_2018_2019_convention
from nnformer.dataset_conversion.Task043_BraTS_2019 import copy_BraTS_segmentation_and_convert_labels
from nnformer.evaluation.region_based_evaluation import get_brats_regions, evaluate_regions
from nnformer.paths import nnFormer_raw_data
import SimpleITK as sitk
import shutil
from medpy.metric import dc, hd95

from nnformer.postprocessing.consolidate_postprocessing import collect_cv_niftis
from typing import Tuple
import random
random.seed(0)

if __name__ == '__main__':
    task_name = "Task999_BraTS2021"
    downloaded_data_dir = "/data2/public_data/BraTS2021"

    target_base = join(nnFormer_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTs)

    img_list = []
    patient_names = []
    cur = join(downloaded_data_dir, "imagesTr")
    lar = join(downloaded_data_dir, "labelsTr")
    for p in os.listdir(cur):
        if p[:-12] not in img_list:
            img_list.append(p[:-12])

    # split
    total_cnt = len(img_list)
    # test_num = int(total_cnt*0.2)
    train_num = int(total_cnt*0.8)
    random.shuffle(img_list)

    train_list = img_list[:train_num]
    test_list = img_list[train_num:]


    for p in train_list:
        patdir = join(cur)
        labledir = join(lar)
        t1 = join(patdir, p + "_0000.nii.gz")
        t1c = join(patdir, p + "_0001.nii.gz")
        t2 = join(patdir, p + "_0002.nii.gz")
        flair = join(patdir, p + "_0003.nii.gz")
        seg = join(labledir, p + ".nii.gz")
        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
            isfile(seg)
        ]), "!!Error!!"

        shutil.copy(t1, join(target_imagesTr, p + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesTr, p + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesTr, p + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesTr, p + "_0003.nii.gz"))
        shutil.copy(seg, join(target_labelsTr, p+".nii.gz"))

    for p in test_list:
        patdir = join(cur)
        labledir = join(lar)
        t1 = join(patdir, p + "_0000.nii.gz")
        t1c = join(patdir, p + "_0001.nii.gz")
        t2 = join(patdir, p + "_0002.nii.gz")
        flair = join(patdir, p + "_0003.nii.gz")
        seg = join(labledir, p + ".nii.gz")
        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
            isfile(seg)
        ]), "!!Error!!"

        shutil.copy(t1, join(target_imagesTs, p + "_0000.nii.gz"))
        shutil.copy(t1c, join(target_imagesTs, p + "_0001.nii.gz"))
        shutil.copy(t2, join(target_imagesTs, p + "_0002.nii.gz"))
        shutil.copy(flair, join(target_imagesTs, p + "_0003.nii.gz"))
        shutil.copy(seg, join(target_labelsTs, p+".nii.gz"))


    json_dict = OrderedDict()
    json_dict['name'] = "BraTS2021"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see BraTS2020"
    json_dict['licence'] = "see BraTS2020 license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
        "3": "FLAIR"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing",
    }
    json_dict['numTraining'] = len(train_list)
    json_dict['numTest'] = len(test_list)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             train_list]
    json_dict['test'] = []
    save_json(json_dict, join(target_base, "dataset.json"))

    print(total_cnt)
    print(len(train_list))
    print(len(test_list))