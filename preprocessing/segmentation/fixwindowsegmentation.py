import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, interpolate
from scipy.signal import find_peaks
from tslearn import metrics
from sklearn.preprocessing import StandardScaler
from visualization.matplotly_plot import plot_sdtw, plot_segmented_gyro, plot_segmented_ik

class FixWindowSegmentation:
    def __init__(self, imu_signal, ik_signal, labels, winsize, overlap, start_over=False):

        self.imu_signal = imu_signal
        self.ik_signal = ik_signal
        self.general_labels = labels
        self.updated_general_labels = []
        self.winsize = winsize
        self.overlap = overlap
        self.x = []
        self.y = []
        self.cach_dir = "./caches/segmentation"
        self.start_over = start_over

    def fixsize_sliding_window(self, data):
        step = round(self.overlap * self.winsize)
        numOfChunks = int(((len(data) - self.winsize) / step) + 1)
        if numOfChunks == 0:
            data_out = self.pad_along_axis(data, self.winsize, axis=0)
            data_out = np.expand_dims(data_out, axis=0)
        else:
            data_out = []
            for i in range(0, numOfChunks * step, step):
                data_out.append(data[i:i + self.winsize])
            data_out = np.asarray(data_out)
        return data_out

        # self.y = self.y.reshape([self.y.shape[0] * self.y.shape[1], self.y.shape[2]])
        # self.updated_general_labels = pd.concat(labels, ignore_index=True)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(3, 1, 1)
        # plt.plot(data[:, 0])
        # plt.subplot(3, 1, 2)
        # plt.plot(data[:, 1])
        # plt.subplot(3, 1, 3)
        # plt.plot(data[:, 2])
        # plt.show()
        # plt.figure()
        # for i in range(len(y)):
        #     plt.subplot(3, 1, 1)
        #     plt.plot(y[i][:, 0])
        #     plt.subplot(3, 1, 2)
        #     plt.plot(y[i][:, 1])
        #     plt.subplot(3, 1, 3)
        #     plt.plot(y[i][:, 2])
        # plt.show()

    def run_label_segmentation(self):
        labels_segmented = []
        for i, data in enumerate(self.general_labels):
            label = self.fixsize_sliding_window(data.values)
            labels_segmented.append(label)
        labels_segmented = np.vstack(labels_segmented)
        self.updated_general_labels = labels_segmented
        # labels_segmented = labels_segmented.reshape([labels_segmented.shape[0] * labels_segmented.shape[1], labels_segmented.shape[2]])
        # self.updated_general_labels = pd.DataFrame(labels_segmented, columns=self.general_labels[0].columns)

    def _run_segmentation(self):
        segmentation_file_path = os.path.join(self.cach_dir, "segmentation_"+ str(self.winsize)+".p")
        if not os.path.exists(segmentation_file_path) or self.start_over:
            self.run_imu_segmentation()
            self.run_ik_segmentation()
            self.run_label_segmentation()
            with open(segmentation_file_path, 'wb') as f:
                pickle.dump([self.x, self.y, self.updated_general_labels], f, protocol=pickle.HIGHEST_PROTOCOL)
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
        data_out = []
        for d in data:
            data_out.append(self.pad_along_axis(d, self.winsize, axis=0))
        return data_out







