import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib import animation
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe'


class Plot4LinkageMechanism:
    def __init__(self, data, dof_labels, activity, mode_variation, range_len, wandb_plot):
        '''
        :param data: kinematic 2d
        :param dof_labels: n kinematic degree of freedom
        :param activity: selected activity (e.g. gait)
        :param range_len:
        '''
        self.data = data
        self.dof_labels = dof_labels
        self.activity = activity
        self.mode_variation = mode_variation
        self.range_len = range_len
        self.wandb_plot = wandb_plot



    def prepare_dic_of_data(self):
        pc123 = {'mean': {},
                 'mean+std': {},
                 'mean-std': {},
                 'u+2std PC1': {}, 'u-2std PC1': {},
                 'u+2std PC2': {}, 'u-2std PC2': {},
                 'u+2std PC3': {}, 'u-2std PC3': {}}
        colors = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]

        # for i, kinematic_label in enumerate(self.dof_labels):
        #     len_data = range(i * self.range_len, i * self.range_len + self.range_len)
        #     pc123['mean'][kinematic_label] = {'values': np.mean(self.data[:, :], 0)[len_data]}
        #     pc123['mean']['plot_attribute'] = {'color': 'k', 'linestyle': 'o-'}
        #     pc123['mean+std'][kinematic_label] = {
        #         'values': np.mean(self.data[:, :], 0)[len_data] + np.std(self.data[:, :], 0)[len_data]}
        #     pc123['mean+std']['plot_attribute'] = {'color': 'gray', 'linestyle': 'o-'}
        #     pc123['mean-std'][kinematic_label] = {
        #         'values': np.mean(self.data[:, :], 0)[len_data] - np.std(self.data[:, :], 0)[len_data]}
        #     pc123['mean-std']['plot_attribute'] = {'color': 'gray', 'linestyle': 'o--'}
        #
        #     pc123['u+2std PC1'][kinematic_label] = {'values': self.pc1_std[len_data]}
        #     pc123['u+2std PC1']['plot_attribute'] = {'color': 'b', 'linestyle': 'o-'}
        #     pc123['u-2std PC1'][kinematic_label] = {'values': self.pc1_nstd[len_data]}
        #     pc123['u-2std PC1']['plot_attribute'] = {'color': 'b', 'linestyle': 'o--'}
        #
        #     pc123['u+2std PC2'][kinematic_label] = {'values': self.pc2_std[len_data]}
        #     pc123['u+2std PC2']['plot_attribute'] = {'color': 'r', 'linestyle': 'o-'}
        #     pc123['u-2std PC2'][kinematic_label] = {'values': self.pc2_nstd[len_data]}
        #     pc123['u-2std PC2']['plot_attribute'] = {'color': 'r', 'linestyle': 'o--'}
        #
        #     pc123['u+2std PC3'][kinematic_label] = {'values': self.pc3_std[len_data]}
        #     pc123['u+2std PC3']['plot_attribute'] = {'color': 'g', 'linestyle': 'o-'}
        #     pc123['u-2std PC3'][kinematic_label] = {'values': self.pc3_nstd[len_data]}
        #     pc123['u-2std PC3']['plot_attribute'] = {'color': 'g', 'linestyle': 'o--'}

        pcs_dic = {'mean': {},
                 'mean + std': {},
                 'mean - std': {}}
        for key, value in self.mode_variation.items():
            pcs_dic[key] = {}

        for i, kinematic_label in enumerate(self.dof_labels):
            len_data = range(i * self.range_len, i * self.range_len + self.range_len)
            pcs_dic['mean'][kinematic_label] = {'values': np.mean(self.data[:, :], 0)[len_data]}
            pcs_dic['mean']['plot_attribute'] = {'color': 'k', 'linestyle': 'o-'}
            pcs_dic['mean + std'][kinematic_label] = {
                'values': np.mean(self.data[:, :], 0)[len_data] + np.std(self.data[:, :], 0)[len_data]}
            pcs_dic['mean + std']['plot_attribute'] = {'color': 'gray', 'linestyle': 'o-'}
            pcs_dic['mean - std'][kinematic_label] = {
                'values': np.mean(self.data[:, :], 0)[len_data] - np.std(self.data[:, :], 0)[len_data]}
            pcs_dic['mean - std']['plot_attribute'] = {'color': 'gray', 'linestyle': 'o--'}
            j = 0
            for m, (key, value) in enumerate(self.mode_variation.items()):
                if j % 2 == 0:
                    linestyle = 'o-'
                else:
                    linestyle = 'o--'
                j += 1
                pcs_dic[key][kinematic_label] = {'values': value[len_data]}
                pcs_dic[key]['plot_attribute'] = {'color': colors[math.floor(m/2)], 'linestyle': linestyle}
        return pcs_dic


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

    def animate(self, x_index):
        pc123 = self.pcs
        x0, y0 = x_index, 0
        # print(x_index)/
        for i, (key, value) in enumerate(pc123.items()):
            theta1 = value['pelvis_tilt']['values'][x_index]
            theta2 = value['hip_flexion_r']['values'][x_index]
            try:
                theta3 = value['knee_angle_r']['values'][x_index]
            except:
                theta3 = value['knee_FE_r']['values'][x_index]
            theta4 = value['ankle_angle_r']['values'][x_index]
            linestyle = value['plot_attribute']['linestyle']
            color = value['plot_attribute']['color']
            x1, y1, x1_d, y1_d = self.pelvis_tilt(theta1, self.l1 * (10 - i) / 3)
            x2, y2, x2_d, y2_d = self.hip_flex(theta1, theta2, self.l2)
            x3, y3 = self.knee_flex(theta1, theta2, theta3, self.l2, self.l3)
            x4, y4 = self.ankle_flex(theta1, theta2, theta3, theta4, self.l2, self.l3, self.l4)
            n_rows = int((len(pc123) - 3) / 2)
            if key[0:4]=='mean':
                    # axes[p].hlines(0, -10, 600, linestyles='-.', colors='c')
                    # axes[p].vlines(0, -100, 50, linestyles='-.', colors='c')
                line1.set_data(x0 + x1, y1)
                line2.set_data(x0 + x2, y2)
                    # line3.set_data()
                    # axes[p].plot([x0, x0 + x1], [y0, y1], linestyle, color=color, label=key)
                    # axes[p].plot([x0, x0 + x2], [y0, y2], linestyle, color=color, label=key)
                    # axes[p].plot([x0 + x2, x0 + x3], [y2, y3], linestyle, color=color, label=key)
                    # axes[p].plot([x0 + x3, x0 + x4], [y3, y4], linestyle, color=color, label=key)
            # else:
            #     # axes[p].subplot(n_rows, 1, key[-1])
            #     axes[p].plot([x0, x0 + x1], [y0, y1], linestyle, color=color, label=key)
            #     axes[p].plot([x0, x0 + x2], [y0, y2], linestyle, color=color, label=key)
            #     axes[p].plot([x0 + x2, x0 + x3], [y2, y3], linestyle, color=color, label=key)
            #     axes[p].plot([x0 + x3, x0 + x4], [y3, y4], linestyle, color=color, label=key)

            # x_phase = [str(i) for i in [0, 20, 40, 60, 80, 100]]
            # plt.xlim([-50, 600])
            # plt.xticks(np.arange(0, 600, 100), x_phase)
        return line1, line2

    def animate_plot_mechanism(self, l1, l2, l3, l4):
        self.pcs = self.prepare_dic_of_data()
        self.l1, self.l2, self.l3, self.l4 = l1, l2, l3, l4
        n_rows = int((len(self.pcs ) - 3) / 2)
        global axes, line1, line2, line3
        fig, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=[10, 8])
        line1, = axes[0].plot([], [], lw=2)
        line2, = axes[0].plot([], [], lw=2)
        # line3, = axes[0].plot([], [], lw=2)

        ani = animation.FuncAnimation(fig, self.animate, np.arange(1, 99),
                                      interval=1, blit=True)
        mywriter = animation.FFMpegWriter()
        ani.save('mymovie_linkage.mp4', writer=mywriter)
        plt.show()

        # mywriter = animation.FFMpegWriter()
        # ani.save('mymovie.mp4', writer=mywriter)

    def plot_mechanism_row_basis(self, l1, l2, l3, l4):
        pc123 = self.prepare_dic_of_data()
        plt.figure(figsize=[10,8])
        for x_index in [0, 20, 40, 60, 80, 99]:
            x0, y0 = x_index * 5, 0
            for i, (key, value) in enumerate(pc123.items()):
                theta1 = value['pelvis_tilt']['values'][x_index]
                theta2 = value['hip_flexion_r']['values'][x_index]
                try:
                    theta3 = value['knee_angle_r']['values'][x_index]
                except:
                    theta3 = value['knee_FE_r']['values'][x_index]
                theta4 = value['ankle_angle_r']['values'][x_index]
                linestyle = value['plot_attribute']['linestyle']
                color = value['plot_attribute']['color']
                x1, y1, x1_d, y1_d = self.pelvis_tilt(theta1, l1 * (10 - i) / 3)
                x2, y2, x2_d, y2_d = self.hip_flex(theta1, theta2, l2)
                x3, y3 = self.knee_flex(theta1, theta2, theta3, l2, l3)
                x4, y4 = self.ankle_flex(theta1, theta2, theta3, theta4, l2, l3, l4)
                n_rows = int((len(pc123) - 3)/ 2)
                if key[0:4]=='mean':
                    for p in range(n_rows):
                        plt.subplot(n_rows, 1, p+1)
                        plt.hlines(0, -10, 600, linestyles='-.', colors='c')
                        plt.vlines(0, -100, 50, linestyles='-.', colors='c')
                        plt.plot([x0, x0 + x1], [y0, y1], linestyle, color=color, label=key)
                        plt.plot([x0, x0 + x2], [y0, y2], linestyle, color=color, label=key)
                        plt.plot([x0 + x2, x0 + x3], [y2, y3], linestyle, color=color, label=key)
                        plt.plot([x0 + x3, x0 + x4], [y3, y4], linestyle, color=color, label=key)
                else:
                    plt.subplot(n_rows, 1, key[-1])
                    plt.plot([x0, x0 + x1], [y0, y1], linestyle, color=color, label=key)
                    plt.plot([x0, x0 + x2], [y0, y2], linestyle, color=color, label=key)
                    plt.plot([x0 + x2, x0 + x3], [y2, y3], linestyle, color=color, label=key)
                    plt.plot([x0 + x3, x0 + x4], [y3, y4], linestyle, color=color, label=key)

                x_phase = [str(i) for i in [0, 20, 40, 60, 80, 100]]
                plt.xlim([-50, 600])
                plt.xticks(np.arange(0, 600, 100), x_phase)
        plt.suptitle(self.activity)
        if self.wandb_plot:
            wandb.log({"plot_Linkage_mechanism": [wandb.Image(plt, caption=self.activity)]})
        else:
            plt.show()

