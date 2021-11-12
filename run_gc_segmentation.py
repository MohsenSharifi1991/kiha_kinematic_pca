'''
This function get the segmentation range which were got from matlab file
input: nn_label_segmentation_gc_ik_all.mat
output: output the segmented kinematic and imu based on segmentation range
to get the .mat file run this function "S1_DP_OPENSIM_Prepration_all_kinematic.m" which is located
here: F:\codes\matlab\Matlab DL OPENSIME Prepration
'''
import os
import pandas as pd
import shutil
import opensim as osim
from opensim_processing.opensim_simulation import SimulateData
from utils import read_write
from config import get_config
import pickle
import matplotlib.pyplot as plt

from utils.read_write import write_matfile

config = get_config()

dataset_folder = config['dataset_path']
ik_setup_main_folder = dataset_folder + 'IKSetups/'
ik_result_main_folder = dataset_folder + 'IKResults/'
analyze_setup_main_folder = dataset_folder + 'AnalyzeSetups/'
analyze_result_main_folder = dataset_folder + 'AnalyzeResults/'
imu_main_folder = dataset_folder + 'IMU/'
xsensimu_main_folder = dataset_folder + 'IMUXsens/'
imuosim_main_folder = dataset_folder + 'IMUOsim/'
marker_main_molder = dataset_folder + 'Marker/'
model_main_folder = dataset_folder + 'Models/spinopelvic_kinematic/'
model_file_suffix = '_scaled_pelvis_marker_adjusted_final_imu.osim'
analyze_setup_file_suffix = '_Setup_AnalyzeTool.xml'

dof6_knee = False
if dof6_knee == True:
    model_main_folder = dataset_folder + 'Models/6dofknee/'
    model_file_suffix = '_scaled_adjusted_6dofknee_imu.osim'
    extension_folder = '_6dofknee'
    extension_folder = '_update'
else:
    model_main_folder = dataset_folder + 'Models/spinopelvic_kinematic/'
    model_file_suffix = '_scaled_pelvis_marker_adjusted_final_imu.osim'
    extension_folder = '_flexlumb'

subject_list = ['S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16',
                'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23',
                'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
                'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39']
subject_list = ['S34', 'S35', 'S36', 'S37', 'S38', 'S39']

segmentation_labels = read_write.read_matfile('E:/dataset/kiha/SegmentationIndex/nn_label_segmentation_gc_ik_all.mat')
with open('E:/dataset/kiha/SegmentationIndex/nn_label_segmentation_gc_ik_all_updateS10S21side.pkl', 'rb') as f:
    segmentation_labels = pickle.load(f)

ii = 0
save = False
segments = []
sides = []
activity = 'gait'
for s, subject in enumerate(subject_list):
    print(subject)
    # set the file and folder, if not exist, create
    model_file = model_main_folder + subject + model_file_suffix
    ik_setup_file = ik_setup_main_folder + 'KIHA_IKSetUp.xml'
    analyze_setup_file = analyze_setup_main_folder +activity+'/baseline' + extension_folder + '/' + subject + analyze_setup_file_suffix
    ik_result_subject_baseline = ik_result_main_folder + subject + '/'+activity+'/baseline' + extension_folder
    ik_result_subject_baseline_gc = ik_result_main_folder + subject + '/'+activity+'/baseline' + extension_folder + '_gc'
    analyze_result_subject_baseline = analyze_result_main_folder + subject + '/'+activity + '/baseline' + extension_folder
    analyze_result_subject_baseline_gc = analyze_result_main_folder + subject + '/'+activity + '/baseline' + extension_folder + '_gc'
    xsensimu_result_subject_baseline = xsensimu_main_folder + subject + '/' + activity + '/baseline' + extension_folder
    xsensimu_result_subject_baseline_gc = xsensimu_main_folder + subject + '/' + activity + '/baseline' + extension_folder + '_gc'
    marker_result_subject = marker_main_molder + subject

    if not os.path.exists(ik_result_subject_baseline):
        os.makedirs(ik_result_subject_baseline)

    if os.path.exists(ik_result_subject_baseline_gc):
        shutil.rmtree(ik_result_subject_baseline_gc)
    if not os.path.exists(ik_result_subject_baseline_gc):
        os.makedirs(ik_result_subject_baseline_gc)

    if not os.path.exists(analyze_result_subject_baseline):
        os.makedirs(analyze_result_subject_baseline)
    if not os.path.exists(analyze_result_subject_baseline_gc):
        os.makedirs(analyze_result_subject_baseline_gc)

    if not os.path.exists(xsensimu_result_subject_baseline):
        os.makedirs(xsensimu_result_subject_baseline)
    if os.path.exists(xsensimu_result_subject_baseline_gc):
        shutil.rmtree(xsensimu_result_subject_baseline_gc)
    if not os.path.exists(xsensimu_result_subject_baseline_gc):
        os.makedirs(xsensimu_result_subject_baseline_gc)
    sides = []
    x_signals = []
    y_signals = []
    simulatedata = SimulateData(model_file, ik_setup_file, analyze_setup_file)


    for (dirpath, dirnames, filenames) in os.walk(ik_result_subject_baseline):
        # filenames = filenames[30:]
        for file in filenames:
            if (file.endswith('.mot') and (file[4]=='I' or file[4]=='1')): # include only ik 1 data, since our segmentation data is based on only forward data
                # read kinematic file
                print(file)
                kinematics = read_write.read_opensim_sto_mot(ik_result_subject_baseline + '/' + file)
                end_trial = kinematics['pelvis_tx'].idxmax()+1
                kinematics = kinematics[0:end_trial]
                # read imu file
                xsens_imu_folder = xsensimu_result_subject_baseline + '/' + file[:-7]
                imus = read_write.read_xsens_imus(xsens_imu_folder)
                if dof6_knee:
                    del imus['C7IMU']
                    del imus['T12IMU']
                for i, sensor in enumerate(imus):
                    imus[sensor] = imus[sensor][0:end_trial]
                # get the segmentation range
                segmentation = segmentation_labels['Segmentation_GC'][(segmentation_labels['Subject_Num']==subject) & (segmentation_labels['Trial_Num']==file[0:3])]
                segmentation = segmentation.iloc[0]

                # get segmented imu and ik
                max_length = len(kinematics)
                for c, seg in enumerate(segmentation):
                    segment = seg[0]-1 # matlab is from 1 and python is from 0--> we subtract 1 from all values
                    side = seg[1]
                    if segment[-1] > max_length:
                        segment = range(segment[0], max_length)
                    segments.append(segment)
                    sides.append(seg[1])
                    kinematic = kinematics.iloc[segment]
                    '-----------------------------------------------'
                    # update the side of segmentation for subject 10 and subject 21. Left and right must switch
                    # this itself update the segmentation label as well. Note: no need to update the side seperatetly on segmentation_labels
                    # if subject =='S10' and side == 'L':
                    #     seg[1] = 'R'
                    #     side = 'R'
                    # elif subject =='S10' and side == 'R':
                    #     seg[1] = 'L'
                    #     side = 'L'
                    # segmentation[c, 1] = seg[1]
                    #
                    # half_length = round(kinematic.shape[0]/2)
                    # if subject == 'S21' and side == 'L':
                    #
                    #     if kinematic['knee_angle_l'][0:half_length].max() < kinematic['knee_angle_l'][half_length+1:].max():
                    #         pass
                    #     else:
                    #         seg[1] = 'R'
                    #         side = 'R'
                    # elif subject =='S21' and side == 'R':
                    #     if kinematic['knee_angle_r'][0:half_length].max() < kinematic['knee_angle_r'][half_length+1:].max():
                    #         pass
                    #     else:
                    #         seg[1] = 'L'
                    #         side = 'L'
                    '-----------------------------------------------'
                    imu = {}
                    for _, sensor in enumerate(imus):
                        imu[sensor] = imus[sensor].iloc[segment]

                    # create gc_segmented kinematic folder
                    if not os.path.exists(ik_result_subject_baseline_gc):
                        os.makedirs(ik_result_subject_baseline_gc)
                    # write new gc_segmented kinematic .mot file
                    timeseriesosimtable = read_write.pd_to_osimtimeseriestable(kinematic)
                    osim.STOFileAdapter().write(timeseriesosimtable, ik_result_subject_baseline_gc + '/' + file[:-4] + '_C' + str(
                                                    c) + '.mot')
                    #
                    # create xsens imu folder and write
                    xsensimu_result_subject_baseline_gc_trial = xsensimu_result_subject_baseline_gc + '/' + file[:-7]+ '_C' + str(c)
                    if not os.path.exists(xsensimu_result_subject_baseline_gc_trial):
                        os.makedirs(xsensimu_result_subject_baseline_gc_trial)
                    simulatedata.export_simulated_imu(imu, xsensimu_result_subject_baseline_gc_trial)

                    # fill updated segmentation label
                    segmentation_labels_dic = segmentation_labels[(segmentation_labels['Subject_Num']==subject) & (segmentation_labels['Trial_Num']==file[0:3])].copy()
                    segmentation_labels_dic['trial_num_gc'] = file[:-4] + '_C' + str(c)
                    segmentation_labels_dic['segment_gc'] = [segment]
                    segmentation_labels_dic['side_gc'] = side
                    if ii ==0:
                        segmentation_labels_updated = segmentation_labels_dic.copy()
                    else:
                        segmentation_labels_updated = segmentation_labels_updated.append(segmentation_labels_dic, ignore_index=True)
                    ii = ii+1


if save:
    # segmentation_labels_updated.to_pickle('nn_label_segmentation_gc_ik_all_with_gc.pkl')
    segmentation_labels.to_pickle('nn_label_segmentation_gc_ik_all_updateS10S21side.pkl')
    segmentation_labels_updated.to_pickle('nn_label_segmentation_gc_ik_all_with_gc_updateS10S21side.pkl')