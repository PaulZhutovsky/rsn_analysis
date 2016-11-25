import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, make_scorer
from sklearn.svm import SVC


def balanced_accuracy(y, y_pred, **kwargs):
    sensitivity = recall_score(y, y_pred)
    specificity = np.sum((y == 0) & (y_pred == 0), dtype=np.float) / np.sum(y == 0, dtype=np.float)
    return 0.5 * (sensitivity + specificity)


def build_classifier_svm(data, labels, kernel='linear', class_weight='balanced', **kwargs):
    svm = SVC(kernel=kernel, class_weight=class_weight, probability=True, **kwargs)
    svm.fit(data, labels)
    id_pos_class = svm.classes_ == labels.max()
    return svm, svm.predict_proba(data)[:, id_pos_class]


def build_classifier_lr(data, labels, regularization='l2', **kwargs):
    if (regularization == 'l1') or (regularization == 'l2'):
        log_reg = LogisticRegressionCV(penalty=regularization, Cs=100, cv=10, solver='liblinear', refit=False,
                                       n_jobs=10, verbose=1, class_weight='balanced', **kwargs)
    else:
        # lambda = 1/C:  if C->inf lambda -> 0. So if we want no regularization we need to set C to a high value
        log_reg = LogisticRegression(C=100000000., class_weight='balanced', solver='liblinear', n_jobs=10,
                                     verbose=1, **kwargs)
    log_reg.fit(data, labels)
    return log_reg


def build_classifier_rf(data, labels, **kwargs):
    rf_clf = RandomForestClassifier(**kwargs)
    params_rf = {'n_estimators': np.arange(10, 200, 20), 'max_features': ['sqrt', 'log2', 0.5, 0.75]}
    balanced_acc_scorer = make_scorer(balanced_accuracy)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    grid_search = GridSearchCV(rf_clf, params_rf, scoring=balanced_acc_scorer, cv=cv, refit=True, verbose=1, n_jobs=10)
    grid_search.fit(data, labels)
    return grid_search, grid_search.cv_results_, grid_search.best_params_


def scale_data(train, test):

    scaler = MinMaxScaler(feature_range=(-1, 1))

    # determine max and min values on training set (per feature) (scale training set with it)
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)

    # apply the found parameters to test set (DO NOT compute them again)
    test_scaled = scaler.transform(test)

    return train_scaled, test_scaled


def feature_selection(train, test, y_train, z_thresh=3.5):
    mean_group_1 = train[y_train == 1].mean(axis=0)
    mean_group_2 = train[y_train == 0].mean(axis=0)

    mean_diffs = mean_group_1 - mean_group_2
    z_scores_diffs = (mean_diffs - mean_diffs.mean())/mean_diffs.std()

    chosen_ftrs = np.abs(z_scores_diffs) >= z_thresh
    print 'Features picked: {}/{}'.format(chosen_ftrs.sum(), chosen_ftrs.size)

    return train[:, chosen_ftrs], test[:, chosen_ftrs]


def is_balanced(labels):
    return np.mod(labels.size, labels.sum()) == 0


def get_cv_instance(y_labels, n_iter=1000, test_size=0.2, loo=False):
    if not loo:
        return StratifiedShuffleSplit(n_splits=n_iter, test_size=test_size)

    # actually leave-one-subject-PER-GROUP-out
    # first determine whether data is balanced
    balanced = is_balanced(y_labels)

    if balanced:
        # just take two subjects out and use always two new subjects (do not use all combinations)
        return StratifiedKFold(n_splits=y_labels/2)
    else:
        #  Just create random subparts of your data for the unbalanced case.
        return StratifiedShuffleSplit(test_size=2, n_splits=50)

