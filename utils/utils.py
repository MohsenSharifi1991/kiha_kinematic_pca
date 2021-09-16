import numpy as np
import pandas as pd


def filter_filenames(subject, filenames, segmentation_labels):
    selected_trials = segmentation_labels.trial_number[segmentation_labels.subject_num == subject].unique()
    updated_filenames = [file for file in filenames if file[0:3] in selected_trials]
    return updated_filenames


def organize_phase_index(segmentation_labels):
    '''
    1. organized segmentation label based on consistant phase 1 and 2 across all trials
    :param segmentation_labels:
    :return:
    '''
    for subject in segmentation_labels.subject_num.unique():
        for trial in segmentation_labels.trial_number[segmentation_labels.subject_num == subject].unique():
            trial_data = segmentation_labels[(segmentation_labels.subject_num == subject)
                                             & (segmentation_labels.trial_number == trial)]
            phase_1_min = trial_data.phase_1_index.min()
            phase_2_min = trial_data.phase_2_index.min()
            if phase_1_min < phase_2_min:
                continue
            else:
                # shift data one row up
                phase_2_data = trial_data.loc[:, ['phase_2_index', 'phase_2_value_pt', 'phase_2_value_ll']]
                phase_2_data = phase_2_data.shift(-1, axis=0)
                trial_data.loc[:, ['phase_2_index', 'phase_2_value_pt', 'phase_2_value_ll']] = phase_2_data
                segmentation_labels[(segmentation_labels.subject_num == subject)
                                    & (segmentation_labels.trial_number == trial)] = trial_data

    segmentation_labels = segmentation_labels.dropna()
    segmentation_labels = segmentation_labels.reset_index(drop=True)
    return segmentation_labels


def add_range_to_segmentation_label(segmentation_labels, bound):
    '''
    # 2. generate range based on organized segmentation labels.
    :param segmentation_labels:
    :param bound:
    :return:
    '''
    start = segmentation_labels.phase_1_index.values - bound
    end = segmentation_labels.phase_2_index.values + bound
    segmentation_range = []
    for i in range(len(start)):
        segmentation_range.append(np.arange(int(start[i]), int(end[i])))
    segmentation_labels['segmentation_range'] = segmentation_range
    return segmentation_labels


def add_patient_group_to_segmentation_label(segmentation_labels, config):
    subject_group = pd.DataFrame({'subject': config['train_subjects'], 'pt_stand': config['pt_stand'],
                                  'pt_stand2flexedsit': config['pt_stand2flexedsit']})
    segmentation_labels['pt_stand'] = np.zeros([segmentation_labels.shape[0], 1])
    segmentation_labels['pt_stand2flexedsit'] = np.zeros([segmentation_labels.shape[0], 1])
    for subject in segmentation_labels.subject_num.unique():
        instance_len = len(segmentation_labels.pt_stand[segmentation_labels.subject_num == subject])
        segmentation_labels.pt_stand[segmentation_labels.subject_num == subject] = instance_len * [
            subject_group.pt_stand[subject_group.subject == subject].values]
        segmentation_labels.pt_stand2flexedsit[segmentation_labels.subject_num == subject] = instance_len * [
            subject_group.pt_stand2flexedsit[subject_group.subject == subject].values]
    return segmentation_labels


def get_activity_index_test(test_labels, activity):
    activity_index = []
    if isinstance(test_labels, np.ndarray):
        activity_column = int(np.where(np.isin(test_labels[0][0], ["STS", "Sit Max Flexsion", "Hip Fle-Ext-R", "Hip Fle-Ext-L", "lunge-R Forward", "lunge-L Forward"]) == True)[0])
        for i, label in enumerate(test_labels):
            if np.all(label[:, activity_column] == activity):
                activity_index.append(i)
    else:
        activity_index = test_labels[activity == test_labels['activity']].index.values

    return activity_index
