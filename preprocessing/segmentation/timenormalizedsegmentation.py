from preprocessing.interpolation import InterpolationSignal
import numpy as np


class TimeNormalizedSegmentation:
    def __init__(self, imu_signal, ik_signal, labels, time_normalized_length, start_over=False):

        self.imu_signal = imu_signal
        self.ik_signal = ik_signal
        self.general_labels = labels
        self.updated_general_labels = []
        self.winsize = time_normalized_length
        self.x = []
        self.y = []
        self.cach_dir = "./caches/segmentation"
        self.start_over = start_over
        self.interpolation_signal = InterpolationSignal(self.winsize)

    def run_imu_segmentation(self):
        x_signals_segmented = []
        for i, data in enumerate(self.imu_signal):
            x = self.interpolation_signal.interpolate_signal(data)
            x_signals_segmented.append(x)
        self.x = np.asarray(x_signals_segmented)

    def run_ik_segmentation(self):
        y_signals_segmented = []
        for i, data in enumerate(self.ik_signal):
            y = self.interpolation_signal.interpolate_signal(data)
            y_signals_segmented.append(y)
        self.y = np.asarray(y_signals_segmented)

    def _run_segmentation(self):
        self.run_imu_segmentation()
        self.run_ik_segmentation()
        return self.x, self.y, self.general_labels