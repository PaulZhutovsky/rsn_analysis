"""
Runs the RSN classification

Usage:
    rsn_classification -h | --help
    rsn_classification [--path_labels=<LABELS> --folder_IC=<FOLDER_IC> --save_file=<SAVE_FILE> --standardize --z_thresh=<Z_VAL> --loo --reg=<REG> --rescale]

Options:
    -h --help                   Show this message
    --path_labels=<LABELS>      Path to the labels
                                [default: /data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt]
    --folder_IC=<FOLDER_IC>     Path to the ICs to use for classification
                                [default: /data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN]
    --save_file=<SAVE_FILE>     Save name for the evaluation of classifier
    --standardize               Whether to standardize the networks (per subject) before applying -1, 1 scaling
    --rescale                   Rescale the networks (per subject) to -1, 1 globally (after standardization) before scaling indiviudual voxels
    --z_thresh=<Z_VAL>          What the z-threshold for the feature selection should be [default: 3.5]
    --loo                       Whether to use leave-one-out cross-validaton
    --reg=<REG>                 Which kind of regularization to apply for the meta-classifier. Valid options are:
                                l1, l2, 0 [default: l2]
"""

import numpy as np
import nibabel as nib
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import os.path as osp
from glob import glob
from time import time
from evaluation_classifier import Evaluater
from docopt import docopt


BASE_IC_NAME = 'dr_stage2_ic{:04}.nii.gz'


def get_ic_nums(folder_path):
    """
    Has to be done in this way because some ICs might be missing (numbering might be not sequential)
    :param folder_path:
    :return:
    """
    networks_files = np.array(glob(osp.join(folder_path, 'dr_stage2_ic*.nii.gz')))
    networks_files_no_file_ext = np.char.partition(networks_files, '.')[:, 0]
    ic_names_str = np.char.rpartition(networks_files_no_file_ext, '_')[:, -1]
    id_ics = np.sort(np.char.replace(ic_names_str, 'ic', '').astype(np.int))
    return id_ics


def load_ic(ic_path):
    ic_component = nib.load(ic_path).get_data()
    # intentionally convert to float64 to be sure that we have enough bits for the processes along the way
    return ic_component.astype(np.float64)


def build_classifier_svm(data, labels, kernel='linear', class_weight='balanced', **kwargs):
    svm = SVC(kernel=kernel, class_weight=class_weight, **kwargs)
    svm.fit(data, labels)
    return svm, svm.decision_function(data)


def build_classifier_lr(data, labels, regularization='l2', **kwargs):
    if (regularization == 'l1') or (regularization == 'l2'):
        log_reg = LogisticRegressionCV(penalty=regularization, Cs=100, cv=10, solver='liblinear', refit=False,
                                       n_jobs=10, verbose=1, class_weight='balanced', **kwargs)
    else:
        log_reg = LogisticRegression(C=0., class_weight='balanced', solver='linlinear', n_jobs=10, verbose=1, **kwargs)
    log_reg.fit(data, labels)
    return log_reg


def scale_data(train, test):

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


def mask_data(ic_network, mask, standardize_network=False, min_max_network=False):
    # after the transpose we have a num_subj x num_voxels_within_brain matrix
    tmp = ic_network[mask, :].T

    # Idea proposed by Rajat: standardize each network for each subject individually before normalizing the features
    if standardize_network:
        tmp = (tmp - tmp.mean(axis=1)[:, np.newaxis]) / tmp.std(axis=1)[:, np.newaxis]

    # 2nd idea proposed by Rajat: rescale the networks per subject to be exactly in the same range (-1, 1)
    if min_max_network:
        # 0-1 scale
        tmp = (tmp - tmp.min(axis=1)[:, np.newaxis]) / (tmp.max(axis=1)[:, np.newaxis] - tmp.min(axis=1)[:, np.newaxis])
        # -1-1 scale
        tmp = tmp * 2. - 1.

    return tmp


def feature_selection(train, test, y_train, z_thresh=3.5):
    mean_group_1 = train[y_train.astype('bool')].mean(axis=0)
    mean_group_2 = train[~y_train.astype('bool')].mean(axis=0)

    mean_diffs = mean_group_1 - mean_group_2
    z_scores_diffs = (mean_diffs - mean_diffs.mean())/mean_diffs.std()

    chosen_ftrs = np.abs(z_scores_diffs) >= z_thresh
    print 'Features picked: {}/{}'.format(chosen_ftrs.sum(), chosen_ftrs.size)

    return train[:, chosen_ftrs], test[:, chosen_ftrs]


def get_cv_instance(y_labels, n_iter=1000, test_size=0.2, loo=False):
    if loo:
        return LeaveOneOut(y_labels.size)
    else:
        return StratifiedShuffleSplit(y=y_labels, n_iter=n_iter, test_size=test_size)


def perform_cross_validation(y_labels, cv, ic_to_take, folder_ic, evaluator, standardize_network=False,
                             z_thresh=3.5, regularization='l2', min_max_network=False):
    n_iter = len(cv)
    num_ic = len(ic_to_take)
    num_subj = y_labels.size
    evaluations_metaclf = np.zeros((n_iter, len(evaluator.evaluations)))
    evaluation_svm = np.zeros((n_iter, num_ic, len(evaluator.evaluations)))
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

        label_train, label_test = y_labels[train_index], y_labels[test_index]

        for id_IC, ic_num in enumerate(ic_to_take):
            print
            print "Current IC: {}/{}".format(id_IC + 1, num_ic)

            t1_ic = time()

            ic_component = load_ic(osp.join(folder_ic, BASE_IC_NAME.format(ic_num)))
            # mask and scale the network on subject level if required
            ic_component = mask_data(ic_component, mask, standardize_network=standardize_network,
                                     min_max_network=min_max_network)

            ic_train, ic_test = ic_component[train_index, :], ic_component[test_index, :]

            ic_train, ic_test = feature_selection(ic_train, ic_test, label_train, z_thresh=z_thresh)
            ic_train, ic_test = scale_data(ic_train, ic_test)

            print 'Train Set: max={:.2f}, min={:.2f}'.format(ic_train.max(), ic_train.min())
            print 'Test Set: max(range)=[{:.2f}, {:.2f}], min(range)=[{:.2f}, {:.2f}]'.format(ic_test.max(axis=0).min(),
                                                                                              ic_test.max(),
                                                                                              ic_test.min(),
                                                                                              ic_test.min(axis=0).max())

            svm_clf, data_for_metaclf[train_index, id_IC] = build_classifier_svm(ic_train, label_train)
            data_for_metaclf[test_index, id_IC] = svm_clf.decision_function(ic_test)

            evaluation_svm[id_iter, id_IC, :] = evaluator.evaluate_prediction(y_true=label_test,
                                                                              y_pred=svm_clf.predict(ic_test),
                                                                              y_score=svm_clf.decision_function(ic_test))
            print 'SVM evaluation:'
            evaluator.print_evaluation()

            print 'Time IC: {:.2f}s'.format(time() - t1_ic)

        print 'Time Iteration: {:.2f}m'.format((time() - t1_iter)/60)

        print
        print 'Create Meta-Classifier'
        train_data_meta = data_for_metaclf[train_index, :]
        test_data_meta = data_for_metaclf[test_index, :]

        log_reg = build_classifier_lr(train_data_meta, label_train, regularization=regularization)

        evaluations_metaclf[id_iter, :] = evaluator.evaluate_prediction(y_true=label_test,
                                                                        y_pred=log_reg.predict(test_data_meta),
                                                                        y_score=log_reg.predict_proba(test_data_meta)[:, 1])
        evaluator.print_evaluation()

    return evaluations_metaclf, evaluation_svm


def get_label(labels_path):
    return np.loadtxt(labels_path).astype(np.int)


def set_file_name_eval(file_name):
    if file_name:
        return file_name
    else:
        return 'evaluation_classifier_{}.npz'.format(int(time()))


def main(args):
    (do_loo, folder_ic, labels_path, regularization, save_eval_name,
     min_max_network, standardize_network, z_thresh) = retrieve_parameters(args)

    y_labels = get_label(labels_path=labels_path)
    ics_given = get_ic_nums(folder_path=folder_ic)
    cv_instance = get_cv_instance(y_labels=y_labels, loo=do_loo)
    evaluator = Evaluater(leave_one_out_case=do_loo)

    eval_meta, eval_svm = perform_cross_validation(y_labels=y_labels, cv=cv_instance, ic_to_take=ics_given,
                                                   folder_ic=folder_ic, evaluator=evaluator,
                                                   standardize_network=standardize_network,
                                                   z_thresh=z_thresh, min_max_network=min_max_network,
                                                   regularization=regularization)

    np.savez_compressed(save_eval_name, eval_meta=eval_meta, eval_svm=eval_svm, eval_labels=evaluator.evaluate_labels,
                        ic_labels=ics_given,
                        params_cv={'standardize_network': standardize_network,
                                   'z_thresh': z_thresh,
                                   'loo': do_loo,
                                   'regularization': regularization,
                                   'min_max_network': min_max_network})


def retrieve_parameters(args):
    folder_ic = args['--folder_IC']
    labels_path = args['--path_labels']
    save_eval_name = set_file_name_eval(args['--save_file'])
    rescale_min_max = args['--rescale']
    # global (across voxels) min-max scaling per subject implies standardization in our approach
    if rescale_min_max:
        standardize_networks = True
    else:
        standardize_networks = args['--standardize']
    z_thresh = float(args['--z_thresh'])
    do_loo = args['--loo']
    regularization = args['--reg']
    return (do_loo, folder_ic, labels_path, regularization, save_eval_name,
            rescale_min_max, standardize_networks, z_thresh)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
