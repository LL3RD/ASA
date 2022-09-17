import shutil
from collections import OrderedDict
from copy import deepcopy
from multiprocessing.pool import Pool
from typing import Tuple

import SimpleITK as sitk
import numpy as np
import scipy.stats as ss
from batchgenerators.utilities.file_and_folder_operations import *
from medpy.metric import dc, hd95
from nnformer.evaluation.region_based_evaluation import get_brats_regions, evaluate_regions
from nnformer.paths import nnFormer_raw_data
from nnformer.postprocessing.consolidate_postprocessing import collect_cv_niftis

if __name__ == '__main__':
    task_name = "Task777_MAE"
    root = "/data2/huangjunjia/nnFormer/nnFormer_raw/nnFormer_raw_data/Task777_MAE/labelsTr/"

    target_base = join(nnFormer_raw_data, task_name)

    patient_names = listdir(root)
    # for sub in listdir(root):

    json_dict = OrderedDict()
    json_dict['name'] = "MAE"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "1D"
    json_dict['reference'] = "see ADNI"
    json_dict['licence'] = "see ADNI license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "forwground"
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in
                             patient_names]
    json_dict['test'] = []

    save_json(json_dict, join(target_base, "dataset.json"))