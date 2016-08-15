import numpy as np
import nibabel as nib
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
import os.path as osp
import os
import pandas as pd
from time import time


EVALUATION_FILE_SAVE='evaluation_classifier_{}.npz'.format(int(time()))
EVALUATION_LABELS = ['accuracy', 'AUC', 'F1', 'recall', 'precision', 'sensitivity', 'specificity']
PATH_FILELIST = '/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/.filelist'
FOLDER_IC_DR = '/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN'
BASE_IC_NAME = 'dr_stage2_ic{:04}.nii.gz'
EXPERIMENTAL_SCALING = False
NUM_ICS = 70


def load_IC(IC_path):
    IC_component = nib.load(IC_path).get_data()
    # intentionally convert to float64 to be sure that we have enough bits for the encoding along the way
    return IC_component.astype(np.float64)


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

    # determine mean and variance on training set (scale training set with it)
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)

    # apply the found parameters to test set (DO NOT compute them again)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled


def load_mask(folder_mask=FOLDER_IC_DR):
    mask = nib.load(osp.join(folder_mask, 'mask.nii.gz'))
    return mask.get_data().astype(np.bool)


def mask_data(IC_network, mask):
    return IC_network[mask, :]


def feature_selection(train, test, y_train, z_thresh=3.5):
    mean_group_1 = train[y_train.astype('bool')].mean(axis=0)
    mean_group_2 = train[~y_train.astype('bool')].mean(axis=0)

    mean_diffs = mean_group_1 - mean_group_2
    z_scores_diffs = (mean_diffs - mean_diffs.mean())/mean_diffs.std()

    chosen_ftrs = np.abs(z_scores_diffs) >= z_thresh
    print 'Features picked" {}/{}'.format(chosen_ftrs.sum(), chosen_ftrs.size)

    return train[:, chosen_ftrs], test[:, chosen_ftrs]


def print_evaluation(eval_metrics):
    print 'Accuracy: {:.2f}, AUC: {:.2f}, F1-score: {:.2f}, Recall: {:.2f}, Precision: {:.2f}, Sensitivity: {:.2f}, Specificity {:.2f}'.format(*eval_metrics)


def perform_CV(y_labels, n_iter=1000, test_size=0.2, num_ics=NUM_ICS, evaluation_labels=EVALUATION_LABELS):
    sss = StratifiedShuffleSplit(y=y_labels, n_iter=n_iter, test_size=test_size)
    num_subj = y_labels.size
    evaluations_metaclf = np.zeros((n_iter, len(evaluation_labels)))
    evaluation_svm = np.zeros((n_iter, num_ics, len(evaluation_labels)))
    mask = load_mask()

    for id_iter, (train_index, test_index) in enumerate(sss):
        print
        print "Current Iteration: {}/{}".format(id_iter + 1, len(sss))
        print "Train Set: {} #patients, {} #controls".format(y_labels[train_index].sum(),
                                                             np.sum(y_labels[train_index] == 0))
        print "Test Set: {} #patients, {} #controls".format(y_labels[test_index].sum(),
                                                              np.sum(y_labels[test_index] == 0))
        t1_iter = time()

        data_for_metaclf = np.zeros((num_subj, num_ics))
        svm_clfs = []

        label_train, label_test = y_labels[train_index], y_labels[test_index]

        for id_IC in xrange(num_ics):
            print
            print "Current IC: {}/{}".format(id_IC + 1, num_ics)

            t1_IC = time()

            IC_component = load_IC(osp.join(FOLDER_IC_DR, BASE_IC_NAME.format(id_IC)))
            IC_component = mask_data(IC_component, mask).T

            # make a vector out of the component
            # IC_component = IC_component.reshape((-1, num_subj)).T


            IC_train = IC_component[train_index, :]
            IC_test = IC_component[test_index, :]

            IC_train, IC_test = feature_selection(IC_train, IC_test, label_train)

            IC_train, IC_test = scale_data(IC_train, IC_test)

            print 'Train Set: max={:.2f}, min={:.2f}'.format(IC_train.max(), IC_train.min())
            print 'Test Set: max={:.2f}, min={:.2f}'.format(IC_test.max(), IC_test.min())

            svm_clf, data_for_metaclf[train_index, id_IC] = build_classifier_svm(IC_train, label_train)
            data_for_metaclf[test_index, id_IC] = svm_clf.decision_function(IC_test)

            svm_clfs.append(svm_clf)

            evaluation_svm[id_iter, id_IC, :] = evaluate_prediction(y_true=label_test, y_pred=svm_clf.predict(IC_test),
                                                                    y_score=svm_clf.decision_function(IC_test))
            print 'SVM evaluation:'
            print_evaluation(evaluation_svm[id_iter, id_IC, :])

            print 'Time IC: {:.2f}s'.format(time() - t1_IC)

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


def get_label(filelist_path=PATH_FILELIST):
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


if __name__ == '__main__':
    y_labels = get_label()

    eval_meta, eval_svm, eval_lab = perform_CV(y_labels)
    np.savez_compressed(EVALUATION_FILE_SAVE, eval_meta=eval_meta, eval_svm=eval_svm, eval_labels=eval_lab)
