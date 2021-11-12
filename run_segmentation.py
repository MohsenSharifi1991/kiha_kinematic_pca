# TODO update the explanation
'''
This function get the get the seg index and generate a range based on seg index. Then, it generates segmented IK and IMU and save
input: activity_index_v2_all_table_updatedLL.csv
output: output the segmented kinematic and imu based on segmentation range defined in matlab file
to get the .csv file run this function "data_exploration.m" which is located
here: F:\codes\matlab\opensim_spinopelvic
'''


import shutil
import opensim as osim
import pandas as pd
import os
from opensim_processing.opensim_simulation import SimulateData
from utils import read_write
from config import get_config
from utils.utils import filter_filenames, organize_phase_index, add_range_to_segmentation_label, \
    add_patient_group_to_segmentation_label


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
dof6_knee = True
if dof6_knee == True:
    model_main_folder = dataset_folder + 'Models/6dofknee/'
    model_file_suffix = '_scaled_adjusted_6dofknee_imu.osim'
    extension_folder = '_6dofknee'
    # extension_folder = ''
else:
    model_main_folder = dataset_folder + 'Models/spinopelvic_kinematic/'
    model_file_suffix = '_scaled_pelvis_marker_adjusted_final_imu.osim'
    extension_folder = '_flexlumb'

if os.path.exists('E:/dataset/kiha/SegmentationIndex/activity_index_v2_all_table_updatedLL.csv'):
    segmentation_labels = pd.read_csv('E:/dataset/kiha/SegmentationIndex/activity_index_v2_all_table_updatedLL.csv')
    segmentation_labels = segmentation_labels.sort_values(by='subject_num')

subject_list = segmentation_labels['subject_num'].unique()
activity = 'rom'
segments = []
for s, subject in enumerate(subject_list):
    print(subject)
    # set the file and folder, if not exist, create
    model_file = model_main_folder + subject + model_file_suffix
    ik_setup_file = ik_setup_main_folder + 'KIHA_IKSetUp.xml'
    analyze_setup_file = analyze_setup_main_folder +activity+'/baseline' + extension_folder + '/' + subject + analyze_setup_file_suffix
    ik_result_subject_baseline = ik_result_main_folder + subject + '/'+activity+'/baseline' + extension_folder
    ik_result_subject_baseline_seg = ik_result_main_folder + subject + '/'+activity+'/baseline' + extension_folder + '_seg'
    analyze_result_subject_baseline = analyze_result_main_folder + subject + '/'+activity + '/baseline' + extension_folder
    analyze_result_subject_baseline_seg = analyze_result_main_folder + subject + '/'+activity + '/baseline' + extension_folder + '_seg'
    xsensimu_result_subject_baseline = xsensimu_main_folder + subject + '/' + activity + '/baseline' + extension_folder
    xsensimu_result_subject_baseline_seg = xsensimu_main_folder + subject + '/' + activity + '/baseline' + extension_folder + '_seg'
    marker_result_subject = marker_main_molder + subject + '/' + activity


    if not os.path.exists(ik_result_subject_baseline):
        os.makedirs(ik_result_subject_baseline)

    if os.path.exists(ik_result_subject_baseline_seg):
        shutil.rmtree(ik_result_subject_baseline_seg)
    if not os.path.exists(ik_result_subject_baseline_seg):
        os.makedirs(ik_result_subject_baseline_seg)

    if not os.path.exists(analyze_result_subject_baseline):
        os.makedirs(analyze_result_subject_baseline)

    if os.path.exists(analyze_result_subject_baseline_seg):
        shutil.rmtree(analyze_result_subject_baseline_seg)
    if not os.path.exists(analyze_result_subject_baseline_seg):
        os.makedirs(analyze_result_subject_baseline_seg)

    if not os.path.exists(xsensimu_result_subject_baseline):
        os.makedirs(xsensimu_result_subject_baseline)

    if os.path.exists(xsensimu_result_subject_baseline_seg):
        shutil.rmtree(xsensimu_result_subject_baseline_seg)
    if not os.path.exists(xsensimu_result_subject_baseline_seg):
        os.makedirs(xsensimu_result_subject_baseline_seg)
    sides = []
    x_signals = []
    y_signals = []
    simulatedata = SimulateData(model_file, ik_setup_file, analyze_setup_file)

    for (dirpath, dirnames, filenames) in os.walk(ik_result_subject_baseline):
        # update file name based on segmentation labels trials
        filenames = filter_filenames(subject, filenames, segmentation_labels)

        for file in filenames:
            if file.endswith('.mot'):
                # read kinematic file
                kinematics = read_write.read_opensim_sto_mot(ik_result_subject_baseline + '/' + file)
                # read imu file
                xsens_imu_folder = xsensimu_result_subject_baseline + '/' + file[0:3]
                imus = read_write.read_xsens_imus(xsens_imu_folder)
                for i, sensor in enumerate(imus):
                    imus[sensor] = imus[sensor]
                del imus['C7IMU']
                del imus['T12IMU']
                # get the segmentation range
                segmentation_s = segmentation_labels.segmentation_start[
                    (segmentation_labels['subject_num']==subject) &
                    (segmentation_labels['trial_number']==file[0:3])].values
                segmentation_e = segmentation_labels.segmentation_end[
                    (segmentation_labels['subject_num'] == subject) & (
                                segmentation_labels['trial_number'] == file[0:3])].values
                segmentation = [range(segmentation_s[i], segmentation_e[i]) for i in range(len(segmentation_s))]
                # get segmented imu and ik
                for c, seg in enumerate(segmentation):
                    segment = seg
                    segments.append(segment)
                    kinematic = kinematics.iloc[segment]
                    imu = {}
                    for _, sensor in enumerate(imus):
                        imu[sensor] = imus[sensor].iloc[segment]

                    # create gc_segmented kinematic folder
                    if not os.path.exists(ik_result_subject_baseline_seg):
                        os.makedirs(ik_result_subject_baseline_seg)
                    # write new gc_segmented kinematic .mot file
                    timeseriesosimtable = read_write.pd_to_osimtimeseriestable(kinematic)
                    osim.STOFileAdapter().write(timeseriesosimtable, ik_result_subject_baseline_seg + '/' + file[:-4] + '_C' + str(
                                                    c) + '.mot')

                    # create xsens imu folder and write
                    xsensimu_result_subject_baseline_gc_trial = xsensimu_result_subject_baseline_seg + '/' + file[:-7]+ '_C' + str(c)
                    if not os.path.exists(xsensimu_result_subject_baseline_gc_trial):
                        os.makedirs(xsensimu_result_subject_baseline_gc_trial)
                    simulatedata.export_simulated_imu(imu, xsensimu_result_subject_baseline_gc_trial)