import os
import shutil
from config import get_config

config = get_config()

dataset_folder = config['dataset_path']
ik_setup_main_folder = dataset_folder + 'IKSetups/'
ik_result_main_folder = dataset_folder + 'IKResults/'
original_folder = 'baseline'
reference_folder = 'baseline_flexlumb'
original_folder_update = 'baseline_update'


subject_list = ['S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16',
                'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23',
                'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
                'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39']


activity = 'gait'
for s, subject in enumerate(subject_list):
    print(subject)
    ik_result_subject_baseline_original = ik_result_main_folder + subject + '/'+activity+'/' + original_folder
    ik_result_subject_baseline_reference = ik_result_main_folder + subject + '/' + activity + '/' + reference_folder
    ik_result_subject_baseline_original_update = ik_result_main_folder + subject + '/' + activity + '/' + original_folder_update
    if not os.path.exists(ik_result_subject_baseline_original_update):
        os.makedirs(ik_result_subject_baseline_original_update)

    for (dirpath, dirnames, filenames) in os.walk(ik_result_subject_baseline_original):
        for file in filenames:
            if file in os.listdir(ik_result_subject_baseline_reference): # include only ik 1 data, since our segmentation data is based on only forward data
                source_ = ik_result_subject_baseline_original + '/' + file
                target_ = ik_result_subject_baseline_original_update
                shutil.copy2(source_, target_)