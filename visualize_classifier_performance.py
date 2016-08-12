"""
Compute classifier evaluation plots

USAGE:
    visualize_classifier_performance PATH_EVAL FIGURE_PATH
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
from check_rsns import check_folder
from docopt import docopt


def load_eval_file(file_path):
    tmp = np.load(file_path)
    return tmp['eval_svm'], tmp['eval_meta'], tmp['eval_labels']


def make_boxplot(data, ylabel, xlabel, title, file_name):
    sns.boxplot(data=data, showmeans=True)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.ylim([data.min() - 0.2, data.max() + 0.2])
    plt.hlines(0.5, 0, data.shape[1] + 1, colors='r', linewidth=2.0)
    plt.title(title, fontsize=20)
    plt.xticks(np.arange(data.shape[1]), np.arange(data.shape[1]) + 1, fontsize=12)
    plt.savefig(file_name)


def make_hist(data, xlabel, title, file_name, bins=20):
    plt.hist(data, bins=bins)
    plt.xlabel(xlabel, fontsize=18)
    plt.title(title, fontsize=20)
    plt.vlines(0.5, ymin=0, ymax=10000, colors='r', linewidth=2.0)
    plt.ylim([0, 300])
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig(file_name)


def main(args):
    path_eval = args['PATH_EVAL']
    figure_path = args['FIGURE_PATH']
    check_folder(figure_path)
    eval_svm, eval_meta, eval_labels = load_eval_file(path_eval)

    for id_eval, eval in enumerate(eval_labels):

        plt.figure(figsize=(20, 10))
        make_boxplot(eval_svm[..., id_eval], ylabel='distribution {}'.format(eval), xlabel='RSNs', title=eval,
                     file_name=osp.join(figure_path, 'svm_{}.png'.format(eval)))

        plt.figure()
        make_hist(eval_meta[:, id_eval], xlabel=eval, title='meta: {}'.format(eval),
                  file_name=osp.join(figure_path, 'meta_{}.png'.format(eval)))
        plt.close('all')




if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)