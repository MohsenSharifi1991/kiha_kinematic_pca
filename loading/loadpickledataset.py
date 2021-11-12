import os
import pickle
import numpy as np
import random
import pandas as pd


class LoadPickleDataSet:
    def __init__(self, config):
        self.dl_dataset_path = config['dataset_path']
        self.dataset_name = config['dl_dataset']
        self.method = config['method']
        self.selected_data_types = config['selected_data_types']
        self.selected_data_status = config['selected_data_status']
        self.selected_sensors = config['selected_sensors']
        self.selected_opensim_labels = config['selected_opensim_labels']
        self.augmentation_subset = config['augmentation_subset']
        self.dataset = []
        self.selected_y_labels = []
        self.selected_y_values = []
        self.selected_x_labels = []
        self.selected_x_values = []
        self.selected_labels = []

    def load_dataset(self):
        dataset_file = self.dl_dataset_path + 'MiniDataset/spinopelvic_kinematic/' + self.dataset_name
        if os.path.isfile(dataset_file):
            print('file exist')
            with open(dataset_file, 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            print('this dataset is not exist: run run_dataset_prepration.py first')

    def get_y_data(self):
        y = self.dataset['y']
        y_labels = y['labels']
        print(y_labels)
        y_values = y['values']
        if 'kinematic_status' in y_labels.columns.values:
            y_labels = y_labels.rename(columns={'kinematic_status': 'status', 'kinematic_types': 'types'})
        if 'subject_num' in y_labels.columns.values:
            y_labels = y_labels.rename(columns={'subject_num': 'subjects'})
        self.selected_y_labels = y_labels.loc[(y_labels['status'].isin(self.selected_data_status))
                                              & (y_labels['types'].isin(self.selected_data_types))]

        if any("augmented" in t for t in self.selected_data_types) and self.augmentation_subset:
            self.selected_y_labels = self.get_subset_of_data(self.selected_y_labels)

        self.selected_y_labels = self.selected_y_labels.reset_index(drop=True)
        selected_y_indexes = self.selected_y_labels.index.values.tolist()
        # selected_y_indexes = y_labels.loc[(y_labels['status'].isin(self.selected_data_status))
        #                                   & (y_labels['types'].isin(self.selected_data_types))].index.values.tolist()
        print('total dataset length: ', len(selected_y_indexes))
        # wandb.log({'training_data_length': len(selected_y_indexes)})
        if self.selected_opensim_labels == ['all']:
            self.selected_y_values = [y_val.iloc[:, 1:].values for i, y_val in enumerate(y_values) if
                                      i in selected_y_indexes]
        else:
            self.selected_y_values = [y_val[self.selected_opensim_labels].values for i, y_val in enumerate(y_values) if i in selected_y_indexes]
        self.selected_y_labels = self.selected_y_labels.reset_index(drop=True)
        del y

    def concat_x(self, selected_x_values):
        concat_x_values = []
        step = len(self.selected_sensors)
        for i in range(0, len(selected_x_values), step):
            concat_x_values.append(np.concatenate(selected_x_values[i:i+step], axis=1))
        return concat_x_values

    def get_x_data(self):
        x = self.dataset['x']
        x_labels = x['labels']
        x_labels = x_labels.reset_index(drop=True)
        x_values = x['values']
        if 'kinematic_status' in x_labels.columns.values:
            x_labels = x_labels.rename(columns={'kinematic_status': 'status', 'kinematic_types': 'types'})
        if 'subject_num' in x_labels.columns.values:
            x_labels = x_labels.rename(columns={'subject_num': 'subjects'})
        self.selected_x_labels = x_labels.loc[(x_labels['status'].isin(self.selected_data_status))
                                              & (x_labels['types'].isin(self.selected_data_types))
                                              & (x_labels['sensors'].isin(self.selected_sensors))]
        if any("augmented" in t for t in self.selected_data_types) and self.augmentation_subset:
            self.selected_x_labels = self.get_subset_of_data(self.selected_x_labels)

        selected_x_indexes = self.selected_x_labels.index.values.tolist()
        x_values = [x_val.values for i, x_val in enumerate(x_values) if i in selected_x_indexes]
        self.selected_x_values = self.concat_x(x_values)
        self.selected_x_labels = self.selected_x_labels.reset_index(drop=True)
        del x

    def get_subset_of_data(self, label):
        ctrials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        x = random.sample(ctrials, 1)
        x = [4]
        subset_ctrials = ['_' + str(c) for c in x]
        subset_ctrials = '|'.join(subset_ctrials)
        augmentations = self.selected_data_types.copy()
        augmentations.remove('baseline')
        subset_labels = []
        for augmentation in augmentations:
            subset_labels.append(label.loc[
                          (label['types'] == augmentation) & (
                              label['trials'].str.contains(subset_ctrials))])
        subset_labels = pd.concat(subset_labels)
        subset_labels = pd.concat([label[label['types'] == 'baseline'], subset_labels])
        return subset_labels

    def rename_stair_activity(self):
        self.selected_y_labels['activity'][
            (self.selected_y_labels['activity'].str.contains('Stair')) &
            (self.selected_y_labels['trial_num'].str.contains('_1'))] = 'Stair Ascent'
        self.selected_y_labels['activity'][
            (self.selected_y_labels['activity'].str.contains('Stair')) &
            (self.selected_y_labels['trial_num'].str.contains('_2'))] = 'Stair Descent'


    def run_get_dataset(self):
        self.load_dataset()
        self.get_y_data()
        self.get_x_data()
        self.rename_stair_activity()
        selected_labels = self.selected_y_labels
        selected_labels['rotation'] = 'False'
        del self.dataset
        return self.selected_x_values, self.selected_y_values, selected_labels



