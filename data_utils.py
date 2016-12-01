from glob import glob
import os.path as osp
import os
from time import time

import nibabel as nib
import numpy as np
import pandas as pd


def check_folder(folder_path):
    if not osp.exists(folder_path):
        os.makedirs(folder_path)


def get_ic_nums(folder_path):
    """
    Has to be done in this way because some ICs might be missing (numbering might be not sequential)
    :param folder_path:
    :return:
    """
    networks_files = np.array(sorted(glob(osp.join(folder_path, 'dr_stage2_ic*.nii.gz'))))
    # remove .nii.gz
    networks_files_no_file_ext = np.char.partition(networks_files, '.')[:, 0]
    # get ic# (e.g. ic0001) using _right_ partition
    ic_names_str = np.char.rpartition(networks_files_no_file_ext, '_')[:, -1]
    # get the sorted network ids by removing the 'ic' prefix
    id_ics = np.sort(np.char.replace(ic_names_str, 'ic', '').astype(np.int))
    return id_ics


def load_ic(ic_path):
    ic_component = nib.load(ic_path).get_data()
    # intentionally convert to float64 to be sure that we have enough bits for the processes along the way
    return ic_component.astype(np.float64)


def load_mask(folder_mask):
    mask = nib.load(osp.join(folder_mask, 'mask.nii.gz'))
    return mask.get_data().astype(np.bool)


def mask_data(ic_network, mask, standardize_network=False, range_correct=False):
    # after the transpose we have a num_subj x num_voxels_within_brain matrix
    network = ic_network[mask, :].T

    # Idea proposed by Rajat: standardize each network for each subject individually before normalizing the features
    if standardize_network:
        network = (network - network.mean(axis=1, keepdims=True)) / network.std(axis=1, keepdims=True)

    # 2nd idea proposed by Guido/Rajat: rescale the networks per subject to be exactly in the same range (-1, 1)
    if range_correct:
        # 0-1 scale
        network = (network - network.min(axis=1, keepdims=True)) / (network.max(axis=1, keepdims=True) -
                                                                    network.min(axis=1, keepdims=True))
        # -1-1 scale
        network = network * 2. - 1.

    return network


def get_labels_covariates(labels_path):
    df =  pd.read_csv(labels_path)
    y = df.patients.values
    gender = df.gender.values
    return y, gender

def set_file_name_eval(file_name):
    if file_name:
        return file_name
    else:
        return 'evaluation_classifier_{}.npz'.format(int(time()))
