from rsn_classification import main

DATA_PATH = ['/data/pzhutovsky/fMRI_data/Oxytosin_study/preprocessing_new/dual_regression_group_melodic25',
             '/data/pzhutovsky/fMRI_data/Oxytosin_study/preprocessing_new/dual_regression_group_biswal2010',
             '/data/pzhutovsky/fMRI_data/Oxytosin_study/preprocessing_new/dual_regression_group_smith2009']

LABELS = '/data/pzhutovsky/fMRI_data/Oxytosin_study/preprocessing_new/design_dual_regression/labels_gender.csv'
SAVE_FILE = ['/data/pzhutovsky/fMRI_data/Oxytosin_study/results_ml_preprocessing_new/dual_regression_group_melodic25/svm_gender/evaluation.npz',
             '/data/pzhutovsky/fMRI_data/Oxytosin_study/results_ml_preprocessing_new/dual_regression_group_biswal2010/svm_gender/evaluation.npz',
             '/data/pzhutovsky/fMRI_data/Oxytosin_study/results_ml_preprocessing_new/dual_regression_group_smith2009/svm_gender/evaluation.npz']

if __name__ == '__main__':

    for i in xrange(len(DATA_PATH)):
        params = {'--path_labels': LABELS,
                  '--folder_IC': DATA_PATH[i],
                  '--save_file': SAVE_FILE[i],
                  '--rescale': True,
                  '--standardize': True,
                  '--loo': False,
                  '--z_thresh': 3.5,
                  '--indv_clf': 'SVM'}
        main(params)
