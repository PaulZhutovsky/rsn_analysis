"""
Runs the RSN classification

Usage:
    rsn_classification -h | --help
    rsn_classification [--path_labels=<LABELS> --folder_IC=<FOLDER_IC> --save_file=<SAVE_FILE> --standardize --z_thresh=<Z_VAL> --loo]

Options:
    -h --help                   Show this message
    --path_labels=<LABELS>      Path to the labels
                                [default: /data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt]
    --folder_IC=<FOLDER_IC>     Path to the ICs to use for classification
                                [default: /data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN]
    --save_file=<SAVE_FILE>     Save name for the evaluation of classifier
    --standardize               Whether to standardize the networks (per subject) before applying -1, 1 scaling
    --z_thresh=<Z_VAL>          What the z-threshold for the feature selection should be [default: 3.5]
    --loo                       Whether to use leave-one-out cross-validaton

"""

import numpy as np
import nibabel as nib
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
import os.path as osp
from glob import glob
from time import time
from docopt import docopt


EVALUATION_LABELS = ['accuracy', 'AUC', 'F1', 'recall', 'precision',
                     'sensitivity', 'specificity', 'positive predictive value']
BASE_IC_NAME = 'dr_stage2_ic{:04}.nii.gz'


def get_ic_nums(folder_path):
    """
    Has to be done in this way because some ICs might be missing (numbering might be not sequential)
    :param folder_path:
    :return:
    """
    networks_files = np.array(glob(osp.join(folder_path, 'dr_stage2_ic*.nii.gz')))
    networks_files_no_file_ext = np.core.defchararray.partition(networks_files, '.')[:, 0]
    ic_names_str = np.core.defchararray.rpartition(networks_files_no_file_ext, '_')[:, -1]
    id_ics = np.sort(np.core.defchararray.replace(ic_names_str, 'ic', '').astype(np.int))
    return id_ics


def load_ic(ic_path):
    ic_component = nib.load(ic_path).get_data()
    # intentionally convert to float64 to be sure that we have enough bits for the encoding along the way
    return ic_component.astype(np.float64)


def build_classifier_svm(data, labels, kernel='linear', class_weight='balanced', **kwargs):
    svm = SVC(kernel=kernel, class_weight=class_weight, **kwargs)
    svm.fit(data, labels)
    return svm, svm.decision_function(data)


def build_classifier_lr(data, labels, **kwargs):
    log_reg = LogisticRegressionCV(penalty='l1', Cs=100, cv=10, solver='liblinear', refit=False, n_jobs=10, verbose=1,
                                   class_weight='balanced', scoring='roc_auc', **kwargs)
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
    specificity = np.sum((y_true == 0) & (y_pred == 0))/float(np.sum(y_true == 0))
    # noinspection PyTypeChecker
    PPV = np.sum((y_true == 1) & (y_pred == 1))/float(np.sum(y_pred == 1))
    if np.isnan(PPV):
        PPV = 0.

    return [accuracy, auc, f1, recall, precision, sensitivity, specificity, PPV]


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


def mask_data(ic_network, mask, standardize=False):
    tmp = ic_network[mask, :].T

    if standardize:
        # Idea proposed by Rajat: standardize each network for each subject individually before normalizing the features
        tmp = (tmp - tmp.mean(axis=1)[:, np.newaxis]) / tmp.std(axis=1)[:, np.newaxis]

    return tmp


def feature_selection(train, test, y_train, z_thresh=3.5):
    mean_group_1 = train[y_train.astype('bool')].mean(axis=0)
    mean_group_2 = train[~y_train.astype('bool')].mean(axis=0)

    mean_diffs = mean_group_1 - mean_group_2
    z_scores_diffs = (mean_diffs - mean_diffs.mean())/mean_diffs.std()

    chosen_ftrs = np.abs(z_scores_diffs) >= z_thresh
    print 'Features picked: {}/{}'.format(chosen_ftrs.sum(), chosen_ftrs.size)

    return train[:, chosen_ftrs], test[:, chosen_ftrs]


def print_evaluation(eval_metrics):
    print 'Accuracy: {:.2f}, AUC: {:.2f}, F1-score: {:.2f}, Recall: {:.2f}, ' \
          'Precision: {:.2f}, Sensitivity: {:.2f}, Specificity: {:.2f}, PPV: {:.2f}'.format(*eval_metrics)


def get_cv_instance(y_labels, n_iter=1000, test_size=0.2, loo=False):
    if loo:
        return LeaveOneOut(y_labels.size)
    else
        return StratifiedShuffleSplit(y=y_labels, n_iter=n_iter, test_size=test_size)


def perform_cross_validation(y_labels, cv, ic_to_take, folder_ic, evaluation_labels=EVALUATION_LABELS,
                             standardize=False, z_thresh=3.5):
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

        label_train, label_test = y_labels[train_index], y_labels[test_index]

        for id_IC, ic_num in enumerate(ic_to_take):
            print
            print "Current IC: {}/{}".format(id_IC + 1, num_ic)

            t1_ic = time()

            ic_component = load_ic(osp.join(folder_ic, BASE_IC_NAME.format(ic_num)))
            ic_component = mask_data(ic_component, mask, standardize=standardize)

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


def get_label(labels_path):
    return np.loadtxt(labels_path).astype(np.int)


def set_file_name_eval(file_name):
    if file_name:
        return file_name
    else:
        return 'evaluation_classifier_{}.npz'.format(int(time()))


def main(args):
    folder_ic = args['--folder_IC']
    labels_path = args['--path_labels']
    save_eval_name = set_file_name_eval(args['--save_file'])
    standardize_networks = args['--standardize']
    z_thresh = float(args['--z_thresh'])
    do_loo = args['--loo']

    y_labels = get_label(labels_path=labels_path)
    ic_given = get_ic_nums(folder_path=folder_ic)
    cv_instance = get_cv_instance(y_labels=y_labels, loo=do_loo)

    eval_meta, eval_svm, eval_lab = perform_cross_validation(y_labels=y_labels, cv=cv_instance, ic_to_take=ic_given,
                                                             folder_ic=folder_ic, standardize=standardize_networks,
                                                             z_thresh=z_thresh)
    np.savez_compressed(save_eval_name, eval_meta=eval_meta, eval_svm=eval_svm, eval_labels=eval_lab,
                        ic_labels=ic_given, params_cv={'standardize_network': standardize_networks,
                                                       'z_thresh': z_thresh,
                                                       'loo': do_loo})


if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)
