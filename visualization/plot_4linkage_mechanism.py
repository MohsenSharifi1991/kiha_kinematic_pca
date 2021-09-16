import math
import matplotlib.pyplot as plt
import numpy as np

class Plot4LinkageMechanism:
    def __init__(self, data, dof_labels, activity,  pc1_std, pc1_nstd, pc2_std, pc2_nstd, pc3_std, pc3_nstd, range_len):
        '''
        :param data: kinematic 2d
        :param dof_labels: n kinematic degree of freedom
        :param activity: selected activity (e.g. gait)
        :param pc1_std:
        :param pc1_nstd:
        :param pc2_std:
        :param pc2_nstd:
        :param pc3_std:
        :param pc3_nstd:
        :param range_len:
        '''
        self.data = data
        self.dof_labels = dof_labels
        self.activity = activity
        self.pc1_std = pc1_std
        self.pc1_nstd = pc1_nstd
        self.pc2_std = pc2_std
        self.pc2_nstd = pc2_nstd
        self.pc3_std = pc3_std
        self.pc3_nstd = pc3_nstd
        self.range_len = range_len


    def prepare_dic_of_data(self):
        pc123 = {'mean': {},
                 'mean+std': {},
                 'mean-std': {},
                 'u+2std PC1': {}, 'u-2std PC1': {},
                 'u+2std PC2': {}, 'u-2std PC2': {},
                 'u+2std PC3': {}, 'u-2std PC3': {}}
        for i, kinematic_label in enumerate(self.dof_labels):
            len_data = range(i * self.range_len, i * self.range_len + self.range_len)
            pc123['mean'][kinematic_label] = {'values': np.mean(self.data[:, :], 0)[len_data]}
            pc123['mean']['plot_attribute'] = {'color': 'k', 'linestyle': 'o-'}
            pc123['mean+std'][kinematic_label] = {
                'values': np.mean(self.data[:, :], 0)[len_data] + np.std(self.data[:, :], 0)[len_data]}
            pc123['mean+std']['plot_attribute'] = {'color': 'gray', 'linestyle': 'o-'}
            pc123['mean-std'][kinematic_label] = {
                'values': np.mean(self.data[:, :], 0)[len_data] - np.std(self.data[:, :], 0)[len_data]}
            pc123['mean-std']['plot_attribute'] = {'color': 'gray', 'linestyle': 'o--'}

            pc123['u+2std PC1'][kinematic_label] = {'values': self.pc1_std[len_data]}
            pc123['u+2std PC1']['plot_attribute'] = {'color': 'b', 'linestyle': 'o-'}
            pc123['u-2std PC1'][kinematic_label] = {'values': self.pc1_nstd[len_data]}
            pc123['u-2std PC1']['plot_attribute'] = {'color': 'b', 'linestyle': 'o--'}

            pc123['u+2std PC2'][kinematic_label] = {'values': self.pc2_std[len_data]}
            pc123['u+2std PC2']['plot_attribute'] = {'color': 'r', 'linestyle': 'o-'}
            pc123['u-2std PC2'][kinematic_label] = {'values': self.pc2_nstd[len_data]}
            pc123['u-2std PC2']['plot_attribute'] = {'color': 'r', 'linestyle': 'o--'}

            pc123['u+2std PC3'][kinematic_label] = {'values': self.pc3_std[len_data]}
            pc123['u+2std PC3']['plot_attribute'] = {'color': 'g', 'linestyle': 'o-'}
            pc123['u-2std PC3'][kinematic_label] = {'values': self.pc3_nstd[len_data]}
            pc123['u-2std PC3']['plot_attribute'] = {'color': 'g', 'linestyle': 'o--'}
        return pc123


    def pelvis_tilt(self, theta1, l1):
        '''
        :param theta1: pelvis tilt
        :param l1: pelvis length
        :return: position of pelvis link
        '''
        # pelvis tilt
        x11 = l1*math.sin(math.radians(-theta1))
        y11 = l1*math.cos(math.radians(-theta1))
        x1 = 0 + x11
        y1 = 0 + y11
        x1_d = 0 + x11*2
        y1_d = 0 + y11*2
        return x1, y1, x1_d, y1_d


    def hip_flex(self,theta1, theta2, l2):
        '''
        :param theta1: pelvis tilt
        :param theta2: hip flexion
        :param l2: femur length
        :return: position of femur link
        '''
        # hip flexion
        x22 = l2*math.cos(math.radians(90-theta2-theta1))
        y22 = -l2*math.sin(math.radians(90-theta2-theta1))
        x2 = 0 + x22
        y2 = 0 + y22
        x2_d = 0 + x22*2
        y2_d = 0 + y22*2
        return x2, y2, x2_d, y2_d


    def knee_flex(self,theta1, theta2, theta3, l2, l3):
        '''
        :param theta1: pelvis tilt
        :param theta2: hip flexion
        :param theta3: knee flexion
        :param l2: femur length
        :param l3: tibia length
        :return: position of tibia link
        '''
        x33 = l2*math.cos(math.radians(90-theta2-theta1)) + l3*math.cos(math.radians(90-theta2-theta1+theta3))
        y33 = -(l2*math.sin(math.radians(90-theta2-theta1)) + l3*math.sin(math.radians(90-theta2-theta1+theta3)))
        x3 = 0 + x33
        y3 = 0 + y33
        return x3, y3


    def ankle_flex(self, theta1, theta2, theta3, theta4,  l2, l3, l4):
        '''
        :param theta1: pelvis tilt
        :param theta2: hip flexion
        :param theta3: knee flexion
        :param theta4: ankle flexion
        :param l2: femur length
        :param l3: tibia length
        :param l4: foot length
        :return: position of foot link
        '''
        x44 = l2*math.cos(math.radians(90-theta2-theta1)) + l3*math.cos(math.radians(90-theta2-theta1+theta3)) + l4*math.cos(math.radians(90-theta2-theta1+theta3-(90+theta4)))
        y44 = -((l2*math.sin(math.radians(90-theta2-theta1)) + l3*math.sin(math.radians(90-theta2-theta1+theta3))) + l4*math.sin(math.radians(90-theta2-theta1+theta3-(90+theta4))))
        x4 = 0 + x44
        y4 = 0 + y44
        return x4, y4


    def plot_mechanism(self, l1, l2, l3, l4):
        pc123 = self.prepare_dic_of_data()
        plt.figure()
        plt.hlines(0, -10, 600, linestyles='-.', colors='c')
        plt.vlines(0, -100, 50, linestyles='-.', colors='c')
        for x_index in [0, 20, 40, 60, 80, 99]:
            x0, y0 = x_index * 5, 0
            for i, (key, value) in enumerate(pc123.items()):
                theta1 = value['pelvis_tilt']['values'][x_index]
                theta2 = value['hip_flexion_r']['values'][x_index]
                theta3 = value['knee_angle_r']['values'][x_index]
                theta4 = value['ankle_angle_r']['values'][x_index]
                linestyle = value['plot_attribute']['linestyle']
                color = value['plot_attribute']['color']
                x1, y1, x1_d, y1_d = self.pelvis_tilt(theta1, l1 * (10 - i) / 3)
                x2, y2, x2_d, y2_d = self.hip_flex(theta1, theta2, l2)
                x3, y3 = self.knee_flex(theta1, theta2, theta3, l2, l3)
                x4, y4 = self.ankle_flex(theta1, theta2, theta3, theta4, l2, l3, l4)
                plt.plot([x0, x0 + x1], [y0, y1], linestyle, color=color, label=key)
                plt.plot([x0, x0 + x2], [y0, y2], linestyle, color=color, label=key)
                plt.plot([x0 + x2, x0 + x3], [y2, y3], linestyle, color=color, label=key)
                plt.plot([x0 + x3, x0 + x4], [y3, y4], linestyle, color=color, label=key)
        x_phase = [str(i) for i in [0, 20, 40, 60, 80, 100]]
        plt.xlim([-50, 600])
        plt.xticks(np.arange(0, 600, 100), x_phase)
        plt.title(self.activity)
        plt.show()

