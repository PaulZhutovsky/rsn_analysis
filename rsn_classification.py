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
    --rescale                   Rescale the networks (per subject) to -1, 1 globally (after standardization) before
                                scaling indiviudual voxels
    --z_thresh=<Z_VAL>          What the z-threshold for the feature selection should be [default: 3.5]
    --loo                       Whether to use leave-one-out cross-validaton
    --reg=<REG>                 Which kind of regularization to apply for the meta-classifier. Valid options are:
                                l1, l2, 0 [default: l2] (Deprecated)
"""

import os.path as osp
from time import time

import numpy as np
from docopt import docopt

from data_utils import get_ic_nums, load_ic, load_mask, mask_data, get_label, set_file_name_eval
from ml_utils import build_classifier_svm, build_classifier_rf, scale_data, feature_selection, get_cv_instance
from evaluation_classifier import Evaluater

BASE_IC_NAME = 'dr_stage2_ic{:04}.nii.gz'


def perform_cross_validation(y_labels, cv, ic_to_take, folder_ic, evaluator, standardize_network=False,
                             z_thresh=3.5, regularization='l2', range_correct_network=False):
    n_iter = len(cv)
    num_ic = len(ic_to_take)
    num_subj = y_labels.size
    evaluations_metaclf = np.zeros((n_iter, len(evaluator.evaluations)))
    evaluation_svm = np.zeros((n_iter, num_ic, len(evaluator.evaluations)))
    mask = load_mask(folder_ic)
    t1_total = time()

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
                                     range_correct=range_correct_network)

            ic_train, ic_test = ic_component[train_index, :], ic_component[test_index, :]

            ic_train, ic_test = feature_selection(ic_train, ic_test, label_train, z_thresh=z_thresh)
            ic_train, ic_test = scale_data(ic_train, ic_test)

            print 'Train Set: max={:.2f}, min={:.2f}'.format(ic_train.max(), ic_train.min())
            print 'Test Set: max(range)=[{:.2f}, {:.2f}], min(range)=[{:.2f}, {:.2f}]'.format(ic_test.max(axis=0).min(),
                                                                                              ic_test.max(),
                                                                                              ic_test.min(),
                                                                                              ic_test.min(axis=0).max())

            svm_clf, data_for_metaclf[train_index, id_IC] = build_classifier_svm(ic_train, label_train)
            data_for_metaclf[test_index, id_IC] = svm_clf.predict_proba(ic_test)[:, svm_clf.classes_ == 1]

            evaluation_svm[id_iter, id_IC, :] = evaluator.evaluate_prediction(y_true=label_test,
                                                                              y_pred=svm_clf.predict(ic_test),
                                                                              y_score=svm_clf.decision_function(
                                                                                  ic_test))
            print 'SVM evaluation:'
            evaluator.print_evaluation()

            print 'Time IC: {:.2f}s'.format(time() - t1_ic)

        print 'Time Iteration: {:.2f}m'.format((time() - t1_iter)/60)

        print
        print 'Create Meta-Classifier'
        train_data_meta = data_for_metaclf[train_index, :]
        test_data_meta = data_for_metaclf[test_index, :]

        # log_reg = build_classifier_lr(train_data_meta, label_train, regularization=regularization)
        rf_clf =  build_classifier_rf(train_data_meta, label_train)
        evaluations_metaclf[id_iter, :] = evaluator.evaluate_prediction(y_true=label_test,
                                                                        y_pred=rf_clf.predict(test_data_meta),
                                                                        y_score=rf_clf.predict_proba(test_data_meta)[:,
                                                                                rf_clf.classes_ == 1])
        evaluator.print_evaluation()
    print 'Total Time: {:.2f}min'.format((time() - t1_total)/60.)
    return evaluations_metaclf, evaluation_svm


def retrieve_parameters(args):
    folder_ic = args['--folder_IC']
    labels_path = args['--path_labels']
    save_eval_name = set_file_name_eval(args['--save_file'])
    range_correct = args['--rescale']
    # global (across voxels) min-max scaling per subject implies standardization in our approach
    if range_correct:
        standardize_networks = True
    else:
        standardize_networks = args['--standardize']
    z_thresh = float(args['--z_thresh'])
    do_loo = args['--loo']
    regularization = args['--reg']
    return (do_loo, folder_ic, labels_path, regularization, save_eval_name,
            range_correct, standardize_networks, z_thresh)


def main(args):
    (do_loo, folder_ic, labels_path, regularization, save_eval_name,
     range_correct, standardize_network, z_thresh) = retrieve_parameters(args)

    y_labels = get_label(labels_path=labels_path)
    ics_given = get_ic_nums(folder_path=folder_ic)
    cv_instance = get_cv_instance(y_labels=y_labels, loo=do_loo)
    evaluator = Evaluater(leave_one_out_case=do_loo)

    eval_meta, eval_svm = perform_cross_validation(y_labels=y_labels, cv=cv_instance, ic_to_take=ics_given,
                                                   folder_ic=folder_ic, evaluator=evaluator,
                                                   standardize_network=standardize_network,
                                                   z_thresh=z_thresh, range_correct_network=range_correct,
                                                   regularization=regularization)

    np.savez_compressed(save_eval_name, eval_meta=eval_meta, eval_svm=eval_svm, eval_labels=evaluator.evaluate_labels(),
                        ic_labels=ics_given,
                        params_cv={'standardize_network': standardize_network,
                                   'z_thresh': z_thresh,
                                   'loo': do_loo,
                                   'regularization': regularization,
                                   'min_max_network': range_correct})


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
