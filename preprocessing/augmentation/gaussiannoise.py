import numpy as np
import pandas as pd


class GaussianNoise:
    '''
    This class add gaussian noise to training data
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def apply_gaussian_noise(self, x, y, labels):
        labels['gaussian_noise'] = 'False'
        label = labels.copy()
        gauss_noise = np.random.normal(self.mean, self.std, x.shape)
        x_noise = x + gauss_noise
        label['gaussian_noise'] = 'True'
        update_x = np.asarray([x, x_noise])
        update_y = np.asarray([y, y])
        update_labels = pd.concat([labels, label])
        return update_x, update_y, update_labels

    def run_add_noise(self, x, y, labels):
        gauss_noise = np.random.normal(self.mean, self.std, x.shape)
        x_noise = x + gauss_noise
        update_x = np.concatenate([x, x_noise])
        update_y = np.concatenate([y, y])
        update_labels = pd.concat([labels, labels])
        return update_x, update_y, update_labels