"""
Creates the labels for the PTSD dataset

Usage:
    create_labels [--filelist_loc=<FILELIST> --save_loc=<LABELS> ]

Options:
    --filelist_loc=<FILELIST>   File location of the filelist used for dual regression
                                [default: /data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/.filelist]
    --save_loc=<LABELS>         Location where to save the labels to (including file name)
                                [default: ./labels.txt]
"""
import pandas as pd
import os
import numpy as np
from docopt import docopt


def get_label(filelist_path):
    """
    General intuition: check in the .filelist which was used to compute the dual_regression (assuming that the order is
    the same) and extract the information on the P (patient) vs. C (control) folder naming of the data
    Returns label vector

    !PATIENTS WILL BE CODED AS 1 AND CONTROLS AS 0!
    -------
    """
    filelist = pd.read_table(filelist_path, header=None).squeeze()
    filelist = filelist.str.split(os.sep)
    # filter for subject folder which starts with either C (controls) or P (patients) to get the right index
    id_subj_folder = [index for index, sub_folder in enumerate(filelist.loc[0])
                      if sub_folder.startswith('C') or sub_folder.startswith('P')][0]
    subj_folders = filelist.str[id_subj_folder]
    # patients will be coded as 1 and controls as 0
    return subj_folders.str.startswith('P').values.astype(np.int)


def save_labels(file_path, labels):
    np.savetxt(file_path, labels, fmt='%d')


def main(args):
    labels = get_label(args['--filelist_loc'])
    save_labels(args['--save_loc'], labels)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)


