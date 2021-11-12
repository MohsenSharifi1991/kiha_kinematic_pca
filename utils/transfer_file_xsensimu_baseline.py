import os
from distutils.dir_util import copy_tree
import shutil
subject_list = ['S09','S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                'S18', 'S19', 'S20', 'S21', 'S22', 'S23',
                'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
                'S31', 'S32', 'S33', 'S34', 'S35','S36', 'S37', 'S38', 'S39']
# subject_list = ['S13', 'S15', 'S16', 'S17',
#                 'S18', 'S19']

# subject_list = ['S09']
main_dir = "E:/dataset/kiha/IMUXsens/"
xsensimu_source_dir = "/baseline"
xsensimu_target_dir = "/baseline_6dofknee"
activity = 'rom'
subject_list = ['S09','S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                'S18', 'S19', 'S20', 'S21', 'S22', 'S23',
                'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
                'S31', 'S32', 'S33', 'S34', 'S35','S36', 'S37', 'S38', 'S39']
for s, subject in enumerate(subject_list):
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(subject)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    source_ = main_dir + subject + '/' + activity + xsensimu_source_dir
    target_ = main_dir + subject + '/' + activity + xsensimu_target_dir
    if os.path.exists(target_):
        shutil.rmtree(target_)
    shutil.copytree(source_, target_)

    # # remove doublicated trial, gait trial with T07, T07_1, T07_2 -->remove T07: gait
    # list_target = os.listdir(target_)
    # double_trial = [i[:3] for i in list_target if i[-2:]== '_1' or i[-2:]=='_2']
    # for i in list_target:
    #     if len(i)==3 and i in double_trial:
    #         shutil.rmtree(target_+'/'+i)




