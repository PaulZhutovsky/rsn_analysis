# experimental scaling
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_expr_scl.npz --standardize

# standard scaling
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20.npz

# standard scaling + lower z_thresh
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_z_3.npz --z_thresh=3.0

# experimental scaling + lower z_thresh
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_z_3_e.npz --z_thresh=3.0 --standardize

#######
# NEW #
#######	

# range scaling + l2
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_l2_range_scl.npz --standardize --rescale --reg=l2

# range scaling + no_reg
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_no_reg_range_scl.npz --standardize --rescale --reg=0

# standard scaling + no_reg
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_no_reg.npz --reg=0

# experimental scaling + no_reg
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_no_reg_exp_scl.npz --standardize --reg=0


#######
# LOO #
#######

# standard scaling + loo + l2
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_loo_l2.npz --loo --reg=l2 

# standard scaling + loo + no_reg
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_loo_no_reg.npz --loo --reg=0 

# expiermental scaling + loo + l2
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_loo_l2_e.npz --loo --reg=l2 --standardize

# expiermental scaling + loo + no_reg
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_loo_no_reg_e.npz --loo --reg=0 --standardize

# range scaling + loo + l2
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_loo_l2_range_scl.npz --loo --reg=l2 --standardize --rescale

# range scaling + loo + no_reg
python rsn_classification.py --path_labels=/data/pzhutovsky/fMRI_data/Oxytosin_study/ICA_group_linearReg.gica/ptsd_controls.txt --folder_IC=/data/pzhutovsky/fMRI_data/Oxytosin_study/dual_regression_beckmann_RSN_20/ --save_file=dual_regression_beckmann_20/evaluation_classifier_20_loo_no_reg_range_scl.npz --loo --reg=0 --standardize --rescale



