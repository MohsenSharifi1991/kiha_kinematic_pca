import itertools

import numpy as np
from loading.loadpickledataset import LoadPickleDataSet
from preprocessing.augmentation.gaussiannoise import GaussianNoise
from preprocessing.augmentation.imurotation import IMURotation
from preprocessing.filter_opensim import FilterOpenSim
from preprocessing.segmentation.fixwindowsegmentation import FixWindowSegmentation
# from preprocessing.segmentation.gaitcyclesegmentation import GaitCycleSegmentation
from preprocessing.segmentation.timenormalizedsegmentation import TimeNormalizedSegmentation
from preprocessing.segmentation.zeropaddingsegmentation import ZeroPaddingSegmentation
import pandas as pd

class KIHADataSet:
    def __init__(self, config):
        self.config = config
        self.x = []
        self.y = []
        self.labels = []
        self.segmentation_method = config['segmentation_method']
        self.risk_factor_activity = config['risk_factor_activity']
        # if self.config['gc_dataset']:
        #     self.segmentation_method = 'timenormalized'
        self._preprocess()
        self.n_sample = len(self.y)
        self.dataset = []
        self.train_subjects = config['train_subjects']
        self.test_subjects = config['test_subjects']

        # self.winsize = 128
        self.train_dataset = {}
        self.test_dataset = {}

    def _preprocess(self):
        getdata_handler = LoadPickleDataSet(self.config)
        x, y, labels = getdata_handler.run_get_dataset()
        # x, y, labels = self.run_outlier_short_long_sample(x, y, labels, min_sample_length=20, max_sample_length=self.config['target_padding_length'])

        if self.segmentation_method == 'fixedwindow':
            segmentation_handler = FixWindowSegmentation(x, y, labels, winsize=128, overlap=0.5, start_over=True)
            self.x, self.y, self.labels = segmentation_handler._run_segmentation()

        if self.segmentation_method == 'timenormalized':
            segmentation_handler = TimeNormalizedSegmentation(x, y, labels, time_normalized_length=self.config['target_padding_length'], start_over=True)
            self.x, self.y, self.labels = segmentation_handler._run_segmentation()

        if self.segmentation_method == 'zeropadding':
            segmentation_handler = ZeroPaddingSegmentation(x, y, labels, target_padding_length=self.config['target_padding_length'], start_over=True)
            self.x, self.y, self.labels = segmentation_handler._run_segmentation()

        # elif self.segmentation_method == 'gaitcycle':
        #     segmentation_handler = GaitCycleSegmentation(x, y, labels, winsize=128, overlap=0.5, start_over=True)
        #     self.x, self.y, self.labels = segmentation_handler._run_segmentation()

        if self.config['opensim_filter']:
            filteropensim_handler = FilterOpenSim(self.y, lowcut=6, fs=100, order=2)
            self.y = filteropensim_handler.run_lowpass_filter()

        if self.config['rotation']:
            imu_rotation_handler = IMURotation(knom=10)
            self.x, self.y, self.labels = imu_rotation_handler.run_rotation(self.x.copy(), self.y.copy(), self.labels.copy())

        if self.config['gaussian_noise']:
            gaussian_noise_handler = GaussianNoise(0, .05)
            self.x, self.y, self.labels = gaussian_noise_handler.run_add_noise(self.x, self.y, self.labels)
        del x, y, labels

        self.run_dataset_filter_riskfactor()

    def run_outlier_short_long_sample(self, x, y, label, min_sample_length=50, max_sample_length=200):
        sample_to_be_remove = []
        for i, yy in enumerate(y):
            if len(yy)< min_sample_length or len(yy)>max_sample_length:
                sample_to_be_remove.append(i)
        if sample_to_be_remove:
            for index in sorted(sample_to_be_remove, reverse=True):
                del x[index]
                del y[index]
            label = label.drop(index=sample_to_be_remove).reset_index(drop=True)
            return x, y, label
        else:
            return x, y, label


    def run_dataset_filter_riskfactor(self):
        self.labels = self.labels[self.labels.activity.isin(self.risk_factor_activity)]
        dataset_index = self.labels.index.values
        self.x = self.x[dataset_index]
        self.y = self.y[dataset_index]
        self.labels = self.labels.reset_index(drop=True)

    def run_dataset_split(self):
        if set(self.test_subjects).issubset(self.train_subjects):
             train_labels = self.labels[~self.labels['subjects'].isin(self.test_subjects)]
             test_labels = self.labels[(self.labels['subjects'].isin(self.test_subjects))
                                   & (self.labels['status'] == 'imu')
                                   & (self.labels['types'] == 'baseline')]
        else:
             train_labels = self.labels[self.labels['subjects'].isin(self.train_subjects)]
             test_labels = self.labels[(self.labels['subjects'].isin(self.test_subjects))
                                        & (self.labels['status'] == 'imu')
                                        & (self.labels['types'] == 'baseline')]
        print(train_labels['subjects'].unique())
        print(test_labels['subjects'].unique())

        train_index = train_labels.index.values
        test_index = test_labels.index.values
        print('training length', len(train_index))
        print('test length', len(test_index))

        self.train_dataset['x'] = self.x[train_index]
        self.train_dataset['y'] = self.y[train_index]
        self.train_dataset['labels'] = train_labels.reset_index(drop=True)

        self.test_dataset['x'] = self.x[test_index]
        self.test_dataset['y'] = self.y[test_index]
        self.test_dataset['labels'] = test_labels.reset_index(drop=True)
        del train_labels, test_labels
        return self.train_dataset,  self.test_dataset

    def run_combine_train_test_dataset(self):
        kihadataset_train, kihadataset_test = self.run_dataset_split()
        # combine train and test data
        kihadataset_train['x'] = np.concatenate([kihadataset_train['x'], kihadataset_test['x']])
        kihadataset_train['y'] = np.concatenate([kihadataset_train['y'], kihadataset_test['y']])
        kihadataset_train['labels'] = pd.concat([kihadataset_train['labels'], kihadataset_test['labels']]).reset_index(
            drop=True)
        return kihadataset_train

    def filter_data_based_on_side(self, data, selected_side='R'):
        kihadataset_train_all = data.copy()
        kihadataset_train_all['labels']['tka_side'] = len(kihadataset_train_all['labels']) * ['Nan']
        kihadataset_train_all['labels']['tka_side'][kihadataset_train_all['labels']['KNEE Status'] == 'BiTKA'] = 'BiTKA'
        kihadataset_train_all['labels']['tka_side'][(kihadataset_train_all['labels']['KNEE Status'] == 'TKA') &
                                                    (kihadataset_train_all['labels'][
                                                         '1-Knee Implant L'] == 'Yes')] = 'LTKA'
        kihadataset_train_all['labels']['tka_side'][(kihadataset_train_all['labels']['KNEE Status'] == 'TKA') &
                                                    (kihadataset_train_all['labels'][
                                                         '1-Knee Implant R'] == 'Yes')] = 'RTKA'
        # left or side selection method:
        # 1) Left or Right
        selected_side = selected_side
        affected_knee = False
        kihadataset_train_all['y'] = kihadataset_train_all['y'][kihadataset_train_all['labels'][
            (kihadataset_train_all['labels']['side_seg'] == selected_side) |
            (kihadataset_train_all['labels']['side_seg'].isnull())].index.to_list()]
        kihadataset_train_all['labels'] = kihadataset_train_all['labels'][
            (kihadataset_train_all['labels']['side_seg'] == selected_side)|
            (kihadataset_train_all['labels']['side_seg'].isnull())
            ].reset_index(drop=True)
        return kihadataset_train_all
