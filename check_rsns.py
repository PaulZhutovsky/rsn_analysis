"""
Visualize RSN properties.

USAGE:
    check_rsns FOLDER_RSN
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_utils import load_ic
from rsn_classification import BASE_IC_NAME
import os.path as osp
import os
from glob import glob
from docopt import docopt


FILE_PATH = osp.dirname(osp.realpath(__file__))
FIGURE_FOLDER = osp.join(FILE_PATH, 'figures')


def remove_zeros(data):
    return data[:, ~np.all(data == 0, axis=0)]


def reshape_network(rsn_data):
    return np.array([rsn_data[..., i].ravel() for i in xrange(rsn_data.shape[-1])])


def check_folder(folder_path):
    if not osp.exists(folder_path):
        os.makedirs(folder_path)


def get_ic_nums(folder_path):
    networks_files = np.array(glob(osp.join(folder_path, 'dr*.nii.gz')))
    networks_files_no_file_ext = np.core.defchararray.partition(networks_files, '.')[:, 0]
    ic_names_str = np.core.defchararray.rpartition(networks_files_no_file_ext, '_')[:, -1]
    id_ics = np.sort(np.core.defchararray.replace(ic_names_str, 'ic', '').astype(np.int))
    return id_ics


def make_boxplot(data, num_ic, file_name):
    sns.boxplot(data=data)
    plt.xticks(np.arange(data.shape[1]), np.arange(data.shape[1]) + 1, fontsize=12)
    plt.title('IC {}'.format(num_ic), fontsize=20)
    plt.xlabel('Subject ID', fontsize=18)
    plt.ylabel('Distribution across *whole* brain', fontsize=18)
    plt.ylim([data.min() - 5, data.max() + 5])
    plt.savefig(file_name)


def compute_correlations(data):
    corr_data = np.corrcoef(data)
    mask = corr_data == np.tril(corr_data)
    np.putmask(corr_data, mask, np.nan)
    corr_data = corr_data.ravel()
    return corr_data[~np.isnan(corr_data)]


def compute_amount_zeros(data):
    return np.sum(data == 0, axis=0)/float(data.shape[0])


def make_histogram(data, num_ic, file_name, xlabel, bins=40):
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel, fontsize=18)
    plt.title('IC {}'.format(num_ic), fontsize=20)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig(file_name)


def get_ic_filename(ic_num):
    return BASE_IC_NAME.format(ic_num)


def main(args):
    folder_rsn = args['FOLDER_RSN']

    figure_folder = osp.join(FIGURE_FOLDER, folder_rsn)
    check_folder(figure_folder)
    ic_nums = get_ic_nums(folder_rsn)
    correlations_across_subj_for_rsn = []
    amount_of_zeros_across_subj_for_rsn = []

    for id_rsn, ic_num in enumerate(ic_nums):
        print '{}/{}'.format(id_rsn + 1, ic_nums.size)
        path_rsn = osp.join(folder_rsn, get_ic_filename(ic_num))
        IC_network = load_ic(path_rsn)
        IC_network = reshape_network(IC_network)
        IC_network = remove_zeros(IC_network)

        plt.figure(figsize=(20, 10))
        make_boxplot(IC_network.T, ic_num + 1, osp.join(figure_folder, 'boxplot_ic_{}.png'.format(ic_num + 1)))
        plt.figure()
        correlations_across_subj_for_rsn.append(compute_correlations(IC_network))

        make_histogram(correlations_across_subj_for_rsn[id_rsn], ic_num + 1,
                       osp.join(figure_folder, 'corr_ic_{}.png'.format(ic_num + 1)),
                       'Correlations between RSN of subjects')

        plt.figure()
        amount_of_zeros_across_subj_for_rsn.append(compute_amount_zeros(IC_network))
        make_histogram(amount_of_zeros_across_subj_for_rsn[id_rsn], ic_num + 1,
                       osp.join(figure_folder, 'zeros_ic_{}.png'.format(ic_num + 1)),
                       'Proportion of 0s across voxels', bins=100)

        plt.close('all')


    correlations_across_subj_for_rsn = np.concatenate(correlations_across_subj_for_rsn)
    make_histogram(correlations_across_subj_for_rsn, 'all', osp.join(figure_folder, 'corr_all.png'),
                   'Correlations between RSN of subjects', bins=80)

    amount_of_zeros_across_subj_for_rsn = np.concatenate(amount_of_zeros_across_subj_for_rsn)
    make_histogram(amount_of_zeros_across_subj_for_rsn, 'all', osp.join(figure_folder, 'zeros_all.png'),
                   'Proportion of 0s across voxels', bins=200)



if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)