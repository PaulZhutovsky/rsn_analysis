"""
Runs the RSN classification

Usage:
    rsn_classification -h | --help
    rsn_classification [--path_filelist=<FILELIST> | --folder_IC=<FOLDER_IC> | --save_file_name=<SAVE>]

Options:
    -h --help                   Show this message
    --path_filelist=<FILELIST>  Path to the filelist
                                [default: /data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/.filelist]
    --folder_IC=<FOLDER_IC>     Path to the ICs to use for classification
                                [default: /data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN]
    --save_file_name=<SAVE>     Save name for the evaluation of classifier
"""

import numpy as np
import nibabel as nib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
import os.path as osp
import os
from glob import glob
import pandas as pd
from time import time
from docopt import docopt


EVALUATION_LABELS = ['accuracy', 'AUC', 'F1', 'recall', 'precision', 'sensitivity', 'specificity']
BASE_IC_NAME = 'dr_stage2_ic{:04}.nii.gz'
EXPERIMENTAL_SCALING = False


def get_ic_nums(folder_path):
    networks_files = np.array(glob(osp.join(folder_path, 'dr_stage2_ic*.nii.gz')))
    networks_files_no_file_ext = np.core.defchararray.partition(networks_files, '.')[:, 0]
    ic_names_str = np.core.defchararray.rpartition(networks_files_no_file_ext, '_')[:, -1]
    id_ics = np.sort(np.core.defchararray.replace(ic_names_str, 'ic', '').astype(np.int))
    return id_ics


def load_ic(ic_path):
    ic_component = nib.load(ic_path).get_data()
    # intentionally convert to float64 to be sure that we have enough bits for the encoding along the way
    return ic_component.astype(np.float64)


def build_classifier_svm(data, labels, **kwargs):
    svm = SVC(**kwargs)
    # svm = LinearSVC(penalty='l1', loss='logistic_regression', dual=False)
    svm.fit(data, labels)
    return svm, svm.decision_function(data)


def build_classifier_lr(data, labels, **kwargs):
    log_reg = LogisticRegressionCV(penalty='l1', cv=10, solver='liblinear', refit=False, n_jobs=10, verbose=1, **kwargs)
    log_reg.fit(data, labels)
    return log_reg


def evaluate_prediction(y_true, y_pred, y_score):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    sensitivity = recall
    # noinspection PyTypeChecker
    specificity = np.sum((y_true == 0) & (y_pred == 0))/float(np.sum(y_pred == 0))

    return [accuracy, auc, f1, recall, precision, sensitivity, specificity]


def scale_data(train, test, experimental=EXPERIMENTAL_SCALING):
    if experimental:
        # Idea proposed by Guido: standardize each network for each subject individually
        train_scaled = (train - train.mean(axis=1)[:, np.newaxis])/train.std(axis=1)[:, np.newaxis]
        test_scaled = (test - test.mean(axis=1)[:, np.newaxis])/test.std(axis=1)[:, np.newaxis]
        return train_scaled, test_scaled

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # determine max and min values on training set (per feature) (scale training set with it)
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)

    # apply the found parameters to test set (DO NOT compute them again)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled


def load_mask(folder_mask):
    mask = nib.load(osp.join(folder_mask, 'mask.nii.gz'))
    return mask.get_data().astype(np.bool)


def mask_data(ic_network, mask):
    return ic_network[mask, :]


def feature_selection(train, test, y_train, z_thresh=3.5):
    mean_group_1 = train[y_train.astype('bool')].mean(axis=0)
    mean_group_2 = train[~y_train.astype('bool')].mean(axis=0)

    mean_diffs = mean_group_1 - mean_group_2
    z_scores_diffs = (mean_diffs - mean_diffs.mean())/mean_diffs.std()

    chosen_ftrs = np.abs(z_scores_diffs) >= z_thresh
    print 'Features picked" {}/{}'.format(chosen_ftrs.sum(), chosen_ftrs.size)

    return train[:, chosen_ftrs], test[:, chosen_ftrs]


def print_evaluation(eval_metrics):
    print 'Accuracy: {:.2f}, AUC: {:.2f}, F1-score: {:.2f}, Recall: {:.2f}, ' \
          'Precision: {:.2f}, Sensitivity: {:.2f}, Specificity {:.2f}'.format(*eval_metrics)


def get_cv_instance(y_labels, n_iter=1000, test_size=0.2):
    return StratifiedShuffleSplit(y=y_labels, n_iter=n_iter, test_size=test_size)


def perform_cross_validation(y_labels, cv, ic_to_take, folder_ic, evaluation_labels=EVALUATION_LABELS):
    n_iter = len(cv)
    num_ic = len(ic_to_take)
    num_subj = y_labels.size
    evaluations_metaclf = np.zeros((n_iter, len(evaluation_labels)))
    evaluation_svm = np.zeros((n_iter, num_ic, len(evaluation_labels)))
    mask = load_mask(folder_ic)

    for id_iter, (train_index, test_index) in enumerate(cv):
        print
        print "Current Iteration: {}/{}".format(id_iter + 1, n_iter)
        print "Train Set: {} #patients, {} #controls".format(y_labels[train_index].sum(),
                                                             np.sum(y_labels[train_index] == 0))
        print "Test Set: {} #patients, {} #controls".format(y_labels[test_index].sum(),
                                                            np.sum(y_labels[test_index] == 0))
        t1_iter = time()

        data_for_metaclf = np.zeros((num_subj, num_ic))
        svm_clfs = []

        label_train, label_test = y_labels[train_index], y_labels[test_index]

        for id_IC, ic_num in enumerate(ic_to_take):
            print
            print "Current IC: {}/{}".format(id_IC + 1, num_ic)

            t1_ic = time()

            ic_component = load_ic(osp.join(folder_ic, BASE_IC_NAME.format(ic_num)))
            ic_component = mask_data(ic_component, mask).T

            ic_train = ic_component[train_index, :]
            ic_test = ic_component[test_index, :]

            ic_train, ic_test = feature_selection(ic_train, ic_test, label_train)

            ic_train, ic_test = scale_data(ic_train, ic_test)

            print 'Train Set: max={:.2f}, min={:.2f}'.format(ic_train.max(), ic_train.min())
            print 'Test Set: max={:.2f}, min={:.2f}'.format(ic_test.max(), ic_test.min())

            svm_clf, data_for_metaclf[train_index, id_IC] = build_classifier_svm(ic_train, label_train)
            data_for_metaclf[test_index, id_IC] = svm_clf.decision_function(ic_test)

            svm_clfs.append(svm_clf)

            evaluation_svm[id_iter, id_IC, :] = evaluate_prediction(y_true=label_test, y_pred=svm_clf.predict(ic_test),
                                                                    y_score=svm_clf.decision_function(ic_test))
            print 'SVM evaluation:'
            print_evaluation(evaluation_svm[id_iter, id_IC, :])

            print 'Time IC: {:.2f}s'.format(time() - t1_ic)

        print 'Time Iteration: {:.2f}m'.format((time() - t1_iter)/60)

        print
        print 'Create Meta-Classifier'
        train_data_meta = data_for_metaclf[train_index, :]
        test_data_meta = data_for_metaclf[test_index, :]

        log_reg = build_classifier_lr(train_data_meta, label_train)

        evaluations_metaclf[id_iter, :] = evaluate_prediction(y_true=label_test, y_pred=log_reg.predict(test_data_meta),
                                                              y_score=log_reg.predict_proba(test_data_meta)[:, 1])
        print_evaluation(evaluations_metaclf[id_iter, :])

    return evaluations_metaclf, evaluation_svm, evaluation_labels


def get_label(filelist_path):
    """
    General intuition: check in the .filelist which was used to compute the dual_regression (assuming that the order is
    the same) and extract the information on the P (patient) vs. C (control) folder naming of the data
    Returns label vector
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


def set_file_name_eval(file_name):
    if file_name:
        return file_name
    else:
        return 'evaluation_classifier_{}.npz'.format(int(time()))


def main(args):
    folder_ic = args['--folder_IC']
    filelist_path = args['--filelist']
    save_eval_name = set_file_name_eval(args['--save_file_name'])

    y_labels = get_label(filelist_path=filelist_path)
    ic_given = get_ic_nums(folder_path=folder_ic)

    cv_instance = get_cv_instance(y_labels=y_labels)

    eval_meta, eval_svm, eval_lab = perform_cross_validation(y_labels=y_labels, cv=cv_instance, ic_to_take=ic_given,
                                                             folder_ic=folder_ic)
    np.savez_compressed(save_eval_name, eval_meta=eval_meta, eval_svm=eval_svm, eval_labels=eval_lab)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)