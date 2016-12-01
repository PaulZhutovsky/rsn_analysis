from visualize_classifier_performance import main

DATA = ['/data/pzhutovsky/fMRI_data/Oxytosin_study/results_ml_preprocessing_new/dual_regression_group_melodic25/gp/evaluation.npz',
        '/data/pzhutovsky/fMRI_data/Oxytosin_study/results_ml_preprocessing_new/dual_regression_group_biswal2010/gp/evaluation.npz',
        '/data/pzhutovsky/fMRI_data/Oxytosin_study/results_ml_preprocessing_new/dual_regression_group_smith2009/gp/evaluation.npz']

FIGURE_FOLDER = ['/data/pzhutovsky/fMRI_data/Oxytosin_study/results_ml_preprocessing_new/figure_melodic25_gp',
                 '/data/pzhutovsky/fMRI_data/Oxytosin_study/results_ml_preprocessing_new/figure_biswal2010_gp',
                 '/data/pzhutovsky/fMRI_data/Oxytosin_study/results_ml_preprocessing_new/figure_smith2009_gp']

if __name__ == '__main__':

    for i in xrange(len(DATA)):
        params = {'PATH_EVAL': DATA[i],
                  'FIGURE_PATH': FIGURE_FOLDER[i]}
        main(params)
