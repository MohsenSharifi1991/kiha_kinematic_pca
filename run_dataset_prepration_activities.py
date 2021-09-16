import os
import pickle
import pandas as pd
from pandas import DataFrame
from utils import read_write, filter
from config import get_config
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
import numpy as np
config = get_config()
if config['gc_dataset'] == True:
    folder_extention = '_seg'
else:
    folder_extention = ''

dataset_folder = config['dataset_path']
ik_setup_main_folder = config['dataset_path'] + 'IKSetups/'
ik_result_main_folder = config['dataset_path'] + 'IKResults/'
analyze_setup_main_folder = config['dataset_path'] + 'AnalyzeSetups/'
analyze_result_main_folder = config['dataset_path'] + 'AnalyzeResults/'
imuxsens_main_folder = config['dataset_path'] + 'IMUXsens/'
imuosim_main_folder = config['dataset_path'] + 'IMUOsim/'
marker_main_molder = config['dataset_path'] + 'Marker/'
model_main_folder = config['dataset_path'] + 'Models/'
notesheet_main_folder = config['dataset_path'] + 'NoteSheet/'
segmentation_index = config['dataset_path'] + 'SegmentationIndex/'
dl_labels = segmentation_index + 'activity_index_v2_all_table_updatedLL.csv'
gait_labels = segmentation_index + 'nn_label_segmentation_gc_ik_all_with_gc.pkl'
stair_labels = segmentation_index + 'activity_stair_index_with_gc.pkl'
information_subject_labels = config['dataset_path']+'Informations_Subject.xlsx'

dl_dataset = config['dataset_path'] + 'MiniDataset/spinopelvic_kinematic/'


class ReadIK:
    def __init__(self, config, subject_list, augmentation_methods, activity_list):
        self.config = config
        self.subject_list = subject_list
        self.augmentation_methods = augmentation_methods
        self.activity_list =  activity_list
        self.kinematic_subjects = []
        self.kinematic_activity = []
        self.kinematic_types = []
        self.kinematic_status = []
        self.kinematic_values = []
        self.kinematic_trials = []
        self.kinematic_sides = []

    def read_notesheet(self, subject):
        notesheet_file = notesheet_main_folder + 'Activity_' + subject + '.xlsx'
        notesheet = read_write.read_activity_notesheet(notesheet_file)
        return notesheet

    def cross_check_trial_num(self, notesheet, trial):
        self.activity = notesheet[notesheet['Trial #'] == int(trial[1:3])]['Activity']
        if 'slow' in self.activity.values[0]:
            self.speed = 'slow'
        elif 'fast' in self.activity.values[0]:
            self.speed = 'fast'
        else:
            self.speed = 'normal'

    def find_turn_index(self, kinematic):
        turn_index = kinematic['pelvis_tx'].idxmax()
        return turn_index

    def update_kinematic(self, kinematic):
        # get note sheet and kinematic file, if the trial has turn, save the index of half by getting the maximum pelvis tx,
        # otherwise save the split index as nana and don't update the kinematic file
        if self.activity.values[0] == 'walk' or self.config['gc_dataset'] is True:
            split_index = None
            updt_kinematic = kinematic
        else:
            split_index = int(self.find_turn_index(kinematic))
            updt_kinematic = kinematic[0:split_index]
        return split_index, updt_kinematic

    def read_labels(self):
        # read information subjects
        information_subject = read_write.read_information_subject(information_subject_labels)
        information_subject = information_subject.iloc[:, range(0,20)]
        # read stair labels
        stair_labels_df = pd.read_pickle(stair_labels)
        stair_labels_df = stair_labels_df.rename(
            columns={'segment_gc': 'segment', 'trial_num_gc': 'trial_num_seg', 'side_gc': 'side_seg'})

        # read gait labels
        gait_labels_df = pd.read_pickle(gait_labels)
        gait_labels_df = gait_labels_df.rename(
            columns={'Subject_Num': 'subject_num', 'Trial_Num': 'trial_num',
                     'segment_gc': 'segment', 'trial_num_gc': 'trial_num_seg', 'side_gc': 'side_seg'})
        gait_labels_df['stair_segment'] = None
        gait_labels_df['activity'] = 'Gait'

        # read sts, lunge, hip flex or rom
        lunge_sts_rom_labels_df = pd.read_csv(dl_labels).sort_values(
            by=['subject_num', 'trial_number', 'segmentation_start']).reset_index(drop=True)
        lunge_sts_rom_labels_df = lunge_sts_rom_labels_df.rename(columns={'trial_number': 'trial_num'})
        lunge_sts_rom_labels_df['stair_segment'] = None
        lunge_sts_rom_labels_df['side_seg'] = None
        ranges = []
        trial_num_segs = []
        for i in range(len(lunge_sts_rom_labels_df)):
            seg_start = lunge_sts_rom_labels_df.iloc[i][['segmentation_start']].values[0]
            seg_end = lunge_sts_rom_labels_df.iloc[i][['segmentation_end']].values[0]
            trial_num = lunge_sts_rom_labels_df.iloc[i][['trial_num']].values[0]
            if i == 0:
                c = 0
                trial_num_seg = trial_num + '_IK_C' + str(c)
                c += 1
            else:
                trial_num_past = lunge_sts_rom_labels_df.iloc[i - 1][['trial_num']].values[0]
                if trial_num == trial_num_past:
                    trial_num_seg = trial_num + '_IK_C' + str(c)
                    c += 1
                else:
                    c = 0
                    trial_num_seg = trial_num + '_IK_C' + str(c)
                    c += 1
            ranges.append(np.arange(seg_start, seg_end))
            trial_num_segs.append(trial_num_seg)
        lunge_sts_rom_labels_df['segment'] = ranges
        lunge_sts_rom_labels_df['trial_num_seg'] = trial_num_segs

        # merge all labels
        columns = ['subject_num', 'activity', 'trial_num', 'stair_segment', 'segment', 'trial_num_seg', 'side_seg']
        labels = pd.DataFrame(columns=columns)
        for i in range(len(gait_labels_df)):
            labels = labels.append(gait_labels_df.iloc[i][
                                       ['subject_num', 'activity', 'trial_num', 'stair_segment', 'segment',
                                        'trial_num_seg', 'side_seg']])
        # short trial which we are excluding
        for i in range(len(stair_labels_df)):
            labels = labels.append(stair_labels_df.iloc[i][
                                       ['subject_num', 'activity', 'trial_num', 'stair_segment', 'segment',
                                        'trial_num_seg', 'side_seg']])
        for i in range(len(lunge_sts_rom_labels_df)):
            labels = labels.append(lunge_sts_rom_labels_df.iloc[i][
                                       ['subject_num', 'activity', 'trial_num', 'stair_segment', 'segment',
                                        'trial_num_seg', 'side_seg']])
        labels = labels.reset_index(drop=True)
        information_subject_temp = pd.DataFrame(columns=list(information_subject.columns.values))
        for i in range(len(labels)):
            information_subject_temp = information_subject_temp.append(information_subject[information_subject['Subject #']==labels.iloc[i]['subject_num']])
        information_subject_temp = information_subject_temp.reset_index(drop=True)
        self.labels = pd.concat([labels, information_subject_temp], axis=1)


    def read_ik_all(self):
        kinematic = {}
        self.read_labels()
        self.read_ik_baseline('imu')
        # self.read_ik_baseline('osimimu')
        # self.read_ik_augmented()
        global imu_kinematic_label
        imu_kinematic_label = self.kinematic_labels
        kinematic['labels'] = self.kinematic_labels
        kinematic['values'] = self.kinematic_values
        return kinematic

    def read_ik_baseline(self, status):
        c = 0
        for subject in self.subject_list:
            for activity in self.activity_list:
                notesheet = self.read_notesheet(subject)
                if activity == 'gait' or activity == 'stair':
                    baseline = '/baseline_flexlumb'
                    folder_extention = '_gc'
                else:
                    baseline = '/baseline'
                    folder_extention = '_seg'
                ik_result_subject_baseline = ik_result_main_folder + subject + '/' + activity + baseline+ folder_extention
                for (dirpath, dirnames, filenames) in os.walk(ik_result_subject_baseline):
                    for file in filenames:
                        if file.endswith('.mot'):
                            kinematic = read_write.read_opensim_sto_mot(ik_result_subject_baseline + '/' + file)
                            self.cross_check_trial_num(notesheet, file)
                            self.kinematic_values.append(kinematic[:-1])
                            kinematic_label_temp = self.labels[(self.labels['subject_num']==subject)
                                                               & (self.labels['trial_num_seg']==file[:-4])].copy()
                            kinematic_label_temp['kinematic_status'] = status
                            kinematic_label_temp['kinematic_types'] = 'baseline'
                            kinematic_label_temp['speed'] = self.speed
                            if c ==0:
                                self.kinematic_labels = kinematic_label_temp.copy()
                            else:
                                self.kinematic_labels = self.kinematic_labels.append(kinematic_label_temp)
                            c +=1

    def read_ik_augmented(self):
        for subject in self.subject_list:
            print('/////////////////////////////////////////////////////////////////////////')
            print(subject)

            notesheet = self.read_notesheet(subject)

            ik_result_subject_augmented = ik_result_main_folder + subject + '/augmented'+folder_extention
            for augmentation_method in self.augmentation_methods:
                print('/////////////////////////////////////////////////////////////////////////')
                print(augmentation_method)
                kinematic_augmentation_folder = ik_result_subject_augmented + '/' + augmentation_method
                for (dirpath, dirnames, filenames) in os.walk(kinematic_augmentation_folder):
                    for file in filenames:
                        if file.endswith('.mot'):
                            print(file)
                            kinematic = read_write.read_opensim_sto_mot(kinematic_augmentation_folder + '/' + file)

                            self.cross_check_trial_num(notesheet, file)
                            split_index, kinematic = self.update_kinematic(kinematic)

                            self.kinematic_values.append(kinematic[:-1])
                            self.kinematic_status.append('osimimu')
                            self.kinematic_subjects.append(subject)
                            self.kinematic_types.append('augmented_' + augmentation_method)
                            self.kinematic_trials.append(file.replace('_IK', '')[:-4])
                            self.turn_indexes.append(split_index)

@dataclass
class IMUData:
    imu_values: DataFrame
    imu_status: str
    imu_subjects: str
    imu_types: str
    imu_trials: str
    imu_sensors: str
    # turn_indexes: None

class ReadIMU:
    def __init__(self, config, kinematic_labels, subject_list, augmentation_methods, activity_list):
        self.xsensimu_features = ['freeacc_s_x', 'freeacc_s_y', 'freeacc_s_z', 'gyr_s_x', 'gyr_s_y', 'gyr_s_z']
        self.config = config
        if config['gc_dataset'] == True:
            self.xsensimu_sensor_list = config['osimimu_sensor_list']
        else:
            self.xsensimu_sensor_list = config['xsensimu_sensor_list']
        self.osimimu_sensor_list = config['osimimu_sensor_list']
        self.imu_sensor_list = config['imu_sensor_list']
        self.subject_list = subject_list
        self.activity_list = activity_list
        self.augmentation_methods = augmentation_methods
        self.imu_subjects = []
        self.imu_trials = []
        self.imu_status = []
        self.imu_types = []
        self.imu_values = []
        self.imu_sensors = []
        # self.turn_indexes = list(kinematic_labels['turn_index'].values)
        self.imu_turn_indexes =[]
        self.counter = 0
        self.num_imu_rotation = 10

    def update_imu(self, i, imu):
        # get note sheet and kinematic file, if the trial has turn, save the index of half by getting the maximum pelvis tx,
        # otherwise save the split index as nana and don't update the kinematic file
        turn_index = self.turn_indexes[i]
        if turn_index is None:
            updt_imu = imu
        else:
            updt_imu = imu[0:int(turn_index)]
        return turn_index, updt_imu

    def read_imu_all(self):
        imu = {}
        self.labels = imu_kinematic_label
        self.read_xsens_imu()
        # self.read_osim_imu_baseline()
        # self.read_osim_imu_augmented()
        imu['labels'] = self.imu_labels
        imu['values'] = self.imu_values
        return imu

    def read_xsens_imu(self):
        c = 0
        for subject in self.subject_list:
            for activity in self.activity_list:
                if activity == 'gait' or activity == 'stair':
                    baseline = '/baseline_flexlumb'
                    folder_extention = '_gc'
                else:
                    baseline = '/baseline'
                    folder_extention = '_seg'
                imu_subject = imuxsens_main_folder + subject + '/' + activity + baseline + folder_extention
                print(subject)
                print(activity)
                for (trial_dirpath, trail_dirnames, trialnames) in os.walk(imu_subject):
                    for trial in trail_dirnames:
                        print(trial)
                        if not 'segmented' in trial:

                            imu_baseline_subject_trial = imu_subject + '/' + trial
                            for s, sensor in enumerate(self.xsensimu_sensor_list):
                                if config['gc_dataset']== True:
                                    imu = read_write.read_osim_imu(imu_baseline_subject_trial + '/' + sensor + '.txt')
                                else:
                                    imu = read_write.read_xsens_imu(imu_baseline_subject_trial + '/' + sensor + '.txt')
                                    imu = imu[self.config['xsensimu_features']]
                                # low pas fillter of xsens imu data
                                imu[list(imu.columns.values)] = filter.butter_lowpass_filter(imu.values, lowcut=6, fs=100, order=2)

                                # split_index, imu = self.update_imu(self.counter, imu)
                                self.imu_values.append(imu[:-1]) # this -1 is to equalize the length of sim imu and xsens imu
                                # self.imu_turn_indexes.append(split_index)
                                if 'IK' not in trial:
                                    trial = trial.replace('_C', '_IK_C')
                                imu_label_temp = self.labels[(self.labels['subject_num'] == subject) & (
                                            self.labels['trial_num_seg'] == trial)].copy()
                                imu_label_temp['kinematic_status'] = 'imu'
                                imu_label_temp['kinematic_types'] = 'baseline'
                                imu_label_temp['sensors'] = self.imu_sensor_list[s]
                                if c == 0:
                                    self.imu_labels = imu_label_temp.copy()
                                else:
                                    self.imu_labels = self.imu_labels.append(imu_label_temp)
                                c += 1
                            self.counter = self.counter + 1

    def read_osim_imu_baseline(self):
        for subject in self.subject_list:
            osimimu_subject_baseline = imuosim_main_folder + subject + '/baseline'+folder_extention
            for (trial_dirpath, trail_dirnames, trialnames) in os.walk(osimimu_subject_baseline):
                for trial in trail_dirnames:
                    osimimu_baseline_subject_trial = osimimu_subject_baseline + '/' + trial
                    for s, sensor in enumerate(self.config['osimimu_sensor_list']):
                        imu = read_write.read_osim_imu(osimimu_baseline_subject_trial + '/' + sensor + '.txt')
                        split_index, imu = self.update_imu(self.counter, imu)
                        self.imu_values.append(imu)
                        self.imu_subjects.append(subject)
                        self.imu_sensors.append(self.imu_sensor_list[s])
                        self.imu_status.append('osimimu')
                        self.imu_types.append('baseline')
                        self.imu_trials.append(trial.replace('_IK', ''))
                        self.imu_turn_indexes.append(split_index)
                    self.counter = self.counter + 1

    def read_osim_imu_augmented(self):
        for subject in self.subject_list:
            osimimu_subject_augmented = imuosim_main_folder + subject + '/augmented'+folder_extention
            for augmentation_method in self.augmentation_methods:
                osimimu_augmentation_folder = osimimu_subject_augmented + '/' + augmentation_method
                for (trial_dirpath, trail_dirnames, trialnames) in os.walk(osimimu_augmentation_folder):
                    for trial in trail_dirnames:
                        osimimu_augmented_subject_trial = osimimu_augmentation_folder + '/' + trial
                        for s, sensor in enumerate(self.config['osimimu_sensor_list']):
                            imu = read_write.read_osim_imu(osimimu_augmented_subject_trial + '/' + sensor + '.txt')

                            split_index, imu = self.update_imu(self.counter, imu)
                            self.imu_values.append(imu)
                            self.imu_subjects.append(subject)
                            self.imu_sensors.append(self.imu_sensor_list[s])
                            self.imu_status.append('osimimu')
                            self.imu_types.append('augmented_' + augmentation_method)
                            self.imu_trials.append(trial.replace('_IK', ''))
                            self.imu_turn_indexes.append(split_index)
                        self.counter = self.counter + 1


start_time = time.time()
sensor_list = config['imu_sensor_list']
augmentation_methods = ['magoffset', 'magwarp', 'magwarpoffset', 'timewarp', 'combined_timewarp_magwrap']
subject_list = ["S09", "S10", "S11", "S12", "S13", "S15", "S16", "S17","S18", "S19","S20",
                     "S21","S22", "S23", "S25", "S26", "S27", "S28", "S29",
                      "S30", "S31", "S32", "S33", "S34", "S35", "S36", "S37", "S38", "S39"]
# subject_list = ["S09"]
activity_list = ['gait', 'stair', 'lunge', 'sts', 'rom']
# activity_list = ['gait']
datatype_list = ['imu', 'ik']
read_ik_handler = ReadIK(config, subject_list, augmentation_methods, activity_list)
kinematic = read_ik_handler.read_ik_all()
read_imu_handler = ReadIMU(config, kinematic['labels'], subject_list, augmentation_methods, activity_list)
imu = read_imu_handler.read_imu_all()

dataset = {}
dataset['metadata'] = imu_kinematic_label
dataset['dataset_info'] = [('dataset_name', 'kiha'),
                            ('sensor_list', sensor_list),
                           ('activity_list', activity_list),
                           ('subject_list', subject_list),
                           ('datatype_list', datatype_list)]



# dataset['imu'] = imu['values']
# dataset['ik'] = kinematic['values']

dataset['x'] = imu
dataset['y'] = kinematic
dataset_file = dl_dataset + "".join(activity_list) + '_' + "".join(datatype_list) + '_' + "".join(subject_list) +'_' + str(len(sensor_list)) + 'sensor_speed'+'.p'

if os.path.isfile(dataset_file):
    print('file exist')
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)
else:
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)