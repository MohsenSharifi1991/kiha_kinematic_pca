import numpy as np
import pandas as pd
from transforms3d.axangles import axangle2mat  # for rotation


class IMURotation:
    '''
    This class use to transform imu gyr orientation
    '''
    def __init__(self, knom):
        self.knom = knom

    def apply_rotation(self, x, y=None, label=None):

        if y is not None:
            updt_y = []
            for k in range(self.knom):
                updt_y.append(y)
            updt_y = np.asarray(updt_y)

        if label is not None:
            label['rotation'] = 'False'
            updt_label = pd.DataFrame()
            label_temp = label.copy()
            for k in range(self.knom):
                label_temp['rotation'] = 'True'
                updt_label = updt_label.append([label_temp], ignore_index=True)

        rotated_xs = []
        for j in range(int(x.shape[1] / 6)):
            rotated_x = []
            for k in range(self.knom):
                r_mat = self.get_rotation()
                x_temp = np.hstack([x[:, 6 * j:6 * j + 3], np.matmul(x[:, 6 * j + 3: 6 * j + 6], r_mat)])
                rotated_x.append(x_temp)
            rotated_xs.append(np.asarray(rotated_x))
        updt_x = np.dstack(rotated_xs)

        updt_x = np.concatenate([np.expand_dims(x, axis=0), updt_x])
        updt_y = np.concatenate([np.expand_dims(y, axis=0), updt_y])
        updt_label = label.append(updt_label, ignore_index=True)
        return updt_x, updt_y, updt_label

    def run_rotation(self, x, y=None, labels=None):
        updt_labels = pd.DataFrame()
        updt_y = []
        updt_x = []

        if y is not None:
            for i in range(len(x)):
                for k in range(self.knom):
                    updt_y.append(y[i, :, :])
            updt_y = np.asarray(updt_y)
        if labels is not None:
            labels_temp = labels.copy()
            for i in range(len(x)):
                for k in range(self.knom):
                    labels_temp['rotation'] = 'True'
                    updt_labels = updt_labels.append([labels_temp.iloc[i, :]], ignore_index=True)

        for i in range(len(x)):
            x1 = []
            for j in range(int(x.shape[2]/6)):
                x2 = []
                for k in range(self.knom):
                    r_mat = self.get_rotation()
                    x_temp = np.hstack([x[i, :, 6*j:6*j+3], np.matmul(x[i, :, 6*j+3: 6*j+6], r_mat)])
                    x2.append(x_temp)
                x1.append(np.asarray(x2))
            updt_x.append(np.dstack(x1))

        updt_x = np.concatenate([x, np.vstack(updt_x)])
        updt_y = np.concatenate([y, updt_y])
        updt_labels = labels.append(updt_labels, ignore_index=True)
        return updt_x, updt_y, updt_labels

    def da_rotation(self):
        '''
        # ## 5. Rotation
        # #### Hyperparameters :  N/A
        :return:
        '''
        axis = np.random.uniform(low=-1, high=1, size=self.x.shape[1])
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        return np.matmul(self.x, axangle2mat(axis, angle))

    def get_rotation(self):
        '''
        # ## 5. Rotation
        # #### Hyperparameters :  N/A
        :return:
        '''
        axis = np.random.uniform(low=-1, high=1, size=3)
        angle = np.random.uniform(low=-np.pi, high=np.pi)
        return axangle2mat(axis, angle)


