import os
import pickle
import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, interpolate
from scipy.signal import find_peaks

class ZeroPaddingSegmentation:
    def __init__(self, imu_signal, ik_signal, labels, target_padding_length=None, start_over=False):
        self.imu_signal = imu_signal
        self.ik_signal = ik_signal
        self.general_labels = labels
        self.updated_general_labels = self.general_labels
        self.x = []
        self.y = []
        self.cach_dir = "./caches/segmentation"
        self.start_over = start_over
        if target_padding_length == None:
            # self.target_padding_length = max([len(i) for i in self.imu_signal])
            self.target_padding_length = max([len(i) for i in self.ik_signal])
        else:
            self.target_padding_length = target_padding_length

    def run_imu_segmentation(self):
        x_signals_segmented = []
        for i, data in enumerate(self.imu_signal):
            x = self.pad_along_axis(data, self.target_padding_length, axis=0)
            x_signals_segmented.append(x)
        self.x = np.asarray(x_signals_segmented)

    def run_ik_segmentation(self):
        y_signals_segmented = []
        for i, data in enumerate(self.ik_signal):
            y = self.pad_along_axis(data, self.target_padding_length, axis=0)
            y_signals_segmented.append(y)
        self.y = np.asarray(y_signals_segmented)

    def _run_segmentation(self):
        segmentation_file_path = os.path.join(self.cach_dir, "segmentation_seg" + str(self.target_padding_length) + ".p")
        if not os.path.exists(segmentation_file_path) or self.start_over:
            self.run_imu_segmentation()
            self.run_ik_segmentation()
            with open(segmentation_file_path, 'wb') as f:
                pickle.dump([self.x, self.y, self.updated_general_labels], f, protocol=pickle.HIGHEST_PROTOCOL)
                segmentation_file_path = os.path.join(self.cach_dir, "segmentation_seg" + str(self.target_padding_length) + ".mat")
                # scipy.io.savemat(segmentation_file_path, dict(x=self.x, y=self.y, labels=self.updated_general_labels.to_dict()))
        else:
            with open(segmentation_file_path, 'rb') as f:
                self.x, self.y, self.updated_general_labels = pickle.load(f)
        return self.x, self.y, self.updated_general_labels

    def pad_along_axis(self, array, target_length, axis=0):
        pad_size = target_length - array.shape[axis]
        axis_nb = len(array.shape)
        if pad_size < 0:
            return a
        npad = [(0, 0) for x in range(axis_nb)]
        npad[axis] = (0, pad_size)
        # npad is a tuple of (n_before, n_after) for each dimension
        b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)
        return b

    def zero_pad_data(self, data):

        if type(data) is list:
            data_out = []
            for d in data:
                data_out.append(self.pad_along_axis(d, self.target_padding_length, axis=0))
        else:
            data_out = self.pad_along_axis(data, self.target_padding_length, axis=0)
        return data_out