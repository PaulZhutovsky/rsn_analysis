"""
Runs the RSN classification

Usage:
    rsn_classification -h | --help
    rsn_classification [--indv_clf=<CLF> --path_labels=<LABELS> --folder_IC=<FOLDER_IC> --save_file=<SAVE_FILE> --standardize --z_thresh=<Z_VAL> --loo --rescale]

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
    --indv_clf=<CLF>            Whether to use SVM or GP [default: 'SVM']
"""

import os.path as osp
from time import time

import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from docopt import docopt

from data_utils import get_ic_nums, load_ic, load_mask, mask_data, get_labels_covariates, set_file_name_eval, check_folder
from ml_utils import build_classifier_svm, build_classifier_gp, build_classifier_rf, scale_data, feature_selection, get_cv_instance
from evaluation_classifier import Evaluater

BASE_IC_NAME = 'dr_stage2_ic{:04}.nii.gz'


def perform_cross_validation(y_labels, cv, ics_to_take, folder_ic, evaluator, standardize_network=False,
                             z_thresh=3.5, range_correct_network=False, covariates=None, indv_clf='SVM'):
    n_iter = cv.get_n_splits()
    num_ic = len(ics_to_take)
    num_subj = y_labels.size
    evaluations_metaclf = np.zeros((n_iter, len(evaluator.evaluations)))
    evaluation_indvclf = np.zeros((n_iter, num_ic, len(evaluator.evaluations)))
    mask = load_mask(folder_ic)
    t1_total = time()
    print 'Classifier used for individual RSNs: {}'.format(indv_clf)
    print 'Covariates used: {}'.format(True if covariates is not None else False)

    for id_iter, (train_index, test_index) in enumerate(cv.split(np.zeros_like(y_labels), y_labels)):
        print
        print "Current Iteration: {}/{}".format(id_iter + 1, n_iter)
        print "Train Set: {} #patients, {} #controls".format(y_labels[train_index].sum(),
                                                             np.sum(y_labels[train_index] == 0))
        print "Test Set: {} #patients, {} #controls".format(y_labels[test_index].sum(),
                                                            np.sum(y_labels[test_index] == 0))
        t1_iter = time()

        data_for_metaclf = np.zeros((num_subj, num_ic))

        label_train, label_test = y_labels[train_index], y_labels[test_index]
        results = Parallel(n_jobs=15, verbose=1)(delayed(fit_one_IC)(num_subj, evaluator, folder_ic, ic_num, id_IC,
                                                                     label_test, label_train, mask, num_ic,
                                                                     range_correct_network, standardize_network,
                                                                     test_index, train_index, z_thresh, covariates,
                                                                     indv_clf)
                                                 for id_IC, ic_num in enumerate(ics_to_take))

        for i in xrange(len(results)):
            data_for_metaclf[:, i] = results[i][0]
            evaluation_indvclf[id_iter, i, :] = results[i][1]

        print 'Time Iteration: {:.2f}m'.format((time() - t1_iter)/60)
        print
        print 'Create Meta-Classifier'
        train_data_meta = data_for_metaclf[train_index, :]
        test_data_meta = data_for_metaclf[test_index, :]

        if covariates is not None:
            train_data_meta = np.column_stack((train_data_meta, covariates[train_index]))
            test_data_meta = np.column_stack((test_data_meta, covariates[test_index]))

        rf_clf, grid_search_results_, best_params_ = build_classifier_rf(train_data_meta, label_train)
        evaluations_metaclf[id_iter, :] = evaluator.evaluate_prediction(y_true=label_test,
                                                                        y_pred=rf_clf.predict(test_data_meta),
                                                                        y_score=rf_clf.predict_proba(test_data_meta)[:,1])
        evaluator.print_evaluation()
        print best_params_
    print 'Total Time: {:.2f}min'.format((time() - t1_total)/60.)
    return evaluations_metaclf, evaluation_indvclf


def fit_one_IC(num_subj, evaluator, folder_ic, ic_num, id_IC, label_test, label_train, mask, num_ic,
               range_correct_network, standardize_network, test_index, train_index, z_thresh, covariates=None,
               indv_clf='SVM'):
    data_for_metaclf = np.zeros(num_subj)
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

    if covariates is not None:
        ic_train = np.column_stack((ic_train, covariates[train_index]))
        ic_test = np.column_stack((ic_test, covariates[test_index]))

    print 'Train Set: max={:.2f}, min={:.2f}'.format(ic_train.max(), ic_train.min())
    print 'Test Set: max(range)=[{:.2f}, {:.2f}], min(range)=[{:.2f}, {:.2f}]'.format(ic_test.max(axis=0).min(),
                                                                                      ic_test.max(),
                                                                                      ic_test.min(),
                                                                                      ic_test.min(axis=0).max())
    if indv_clf == 'SVM':
        indv_clf, data_for_metaclf[train_index] = build_classifier_svm(ic_train, label_train)
    elif indv_clf == 'GP':
        indv_clf, data_for_metaclf[train_index] = build_classifier_gp(ic_train, label_train)
    else:
        raise RuntimeError('Only GP and SVM supported so far. {} was used'.format(indv_clf))

    data_for_metaclf[test_index] = indv_clf.predict_proba(ic_test)[:, indv_clf.classes_ == 1]
    evaluation_indv = evaluator.evaluate_prediction(y_true=label_test, y_pred=indv_clf.predict(ic_test),
                                                    y_score=data_for_metaclf[test_index])
    print 'Individual Classifier evaluation:'
    evaluator.print_evaluation()
    print 'Time IC: {:.2f}s'.format(time() - t1_ic)
    return data_for_metaclf, evaluation_indv


def retrieve_parameters(args):
    folder_ic = args['--folder_IC']
    labels_path = args['--path_labels']
    save_eval_name = set_file_name_eval(args['--save_file'])
    save_folder = osp.dirname(save_eval_name)
    check_folder(save_folder)
    range_correct = args['--rescale']
    indv_clf = args['--indv_clf']

    # global (across voxels) min-max scaling per subject implies standardization in our approach
    if range_correct:
        standardize_networks = True
    else:
        standardize_networks = args['--standardize']

    z_thresh = float(args['--z_thresh'])
    do_loo = args['--loo']
    return (do_loo, folder_ic, labels_path, save_eval_name,
            range_correct, standardize_networks, z_thresh, indv_clf)


def main(args):
    (do_loo, folder_ic, labels_path, save_eval_name, range_correct, standardize_network,
     z_thresh, indv_clf) = retrieve_parameters(args)

    y_labels, gender = get_labels_covariates(labels_path=labels_path)
    ics_given = get_ic_nums(folder_path=folder_ic)
    cv_instance = get_cv_instance(y_labels=y_labels, loo=do_loo)
    evaluator = Evaluater(leave_one_out_case=do_loo)

    eval_meta, eval_svm = perform_cross_validation(y_labels=y_labels, cv=cv_instance, ics_to_take=ics_given,
                                                   folder_ic=folder_ic, evaluator=evaluator,
                                                   standardize_network=standardize_network,
                                                   z_thresh=z_thresh, range_correct_network=range_correct,
                                                   covariates=gender, indv_clf=indv_clf)

    np.savez_compressed(save_eval_name, eval_meta=eval_meta, eval_svm=eval_svm, eval_labels=evaluator.evaluate_labels(),
                        ic_labels=ics_given,
                        params_cv={'standardize_network': standardize_network,
                                   'z_thresh': z_thresh,
                                   'loo': do_loo,
                                   'min_max_network': range_correct,
                                   'indv_clf': indv_clf})


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
