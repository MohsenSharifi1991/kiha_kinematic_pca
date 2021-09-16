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

subject_list = ['S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16',
                'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23',
                'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
                'S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39']
subject_list = [ 'S18']

segmentation_labels = read_write.read_matfile('E:/dataset/kiha/SegmentationIndex/nn_label_segmentation_gc_ik_all.mat')
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
    analyze_setup_file = analyze_setup_main_folder + '/'+activity+'/baseline_flexlumb/' + subject + analyze_setup_file_suffix
    ik_result_subject_baseline = ik_result_main_folder + subject + '/'+activity+'/baseline_flexlumb'
    ik_result_subject_baseline_gc = ik_result_main_folder + subject + '/'+activity+'/baseline_flexlumb_gc'
    analyze_result_subject_baseline = analyze_result_main_folder + subject + '/'+activity+'/baseline_flexlumb'
    analyze_result_subject_baseline_gc = analyze_result_main_folder + subject + '/'+activity+'/baseline_flexlumb_gc'
    xsensimu_result_subject_baseline = xsensimu_main_folder + subject + '/'+activity+'/baseline_flexlumb'
    xsensimu_result_subject_baseline_gc = xsensimu_main_folder + subject + '/'+activity+'/baseline_flexlumb_gc'
    marker_result_subject = marker_main_molder + subject

    # if not os.path.exists(ik_result_subject_baseline):
    #     os.makedirs(ik_result_subject_baseline)
    #
    # if os.path.exists(ik_result_subject_baseline_gc):
    #     shutil.rmtree(ik_result_subject_baseline_gc)
    # if not os.path.exists(ik_result_subject_baseline_gc):
    #     os.makedirs(ik_result_subject_baseline_gc)
    #
    # if not os.path.exists(analyze_result_subject_baseline):
    #     os.makedirs(analyze_result_subject_baseline)
    # if not os.path.exists(analyze_result_subject_baseline_gc):
    #     os.makedirs(analyze_result_subject_baseline_gc)
    #
    # if not os.path.exists(xsensimu_result_subject_baseline):
    #     os.makedirs(xsensimu_result_subject_baseline)
    # if os.path.exists(xsensimu_result_subject_baseline_gc):
    #     shutil.rmtree(xsensimu_result_subject_baseline_gc)
    # if not os.path.exists(xsensimu_result_subject_baseline_gc):
    #     os.makedirs(xsensimu_result_subject_baseline_gc)
    sides = []
    x_signals = []
    y_signals = []
    simulatedata = SimulateData(model_file, ik_setup_file, analyze_setup_file)

    for (dirpath, dirnames, filenames) in os.walk(ik_result_subject_baseline):
        filenames = filenames[30:]
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
                    imu = {}
                    for _, sensor in enumerate(imus):
                        imu[sensor] = imus[sensor].iloc[segment]

                    # import matplotlib.pyplot as plt
                    # plt.figure()
                    # plt.plot(kinematic[['hip_flexion_l', 'knee_FE_l']])
                    # plt.show()
                    # plt.figure()
                    # plt.subplot(1, 2, 1)
                    # plt.plot(imu['LFootIMU'].iloc[:, 0:3])
                    # plt.subplot(1, 2, 2)
                    # plt.plot(imu['LFootIMU'].iloc[:, 3:6])
                    # plt.show()


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
    segmentation_labels_updated.to_pickle('nn_label_segmentation_gc_ik_all_with_gc.pkl')