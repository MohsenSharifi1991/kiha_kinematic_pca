import itertools
from venv import logger

import networkx
import scipy
import wandb
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import math
import pandas as pd
import seaborn as sns
from collections import defaultdict
from dtaidistance import dtw
from dtaidistance import clustering
from dtaidistance import ed
from dtaidistance import dtw_visualisation as dtwvis



def plot_imu_osimimu(imu_name, gyr_imu, gyr_osim, acc_imu, acc_osim):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(gyr_imu[:, 0], 'r', label='x imu')
    plt.plot(gyr_imu[:, 1], 'g', label='y imu')
    plt.plot(gyr_imu[:, 2], 'b', label='z imu')
    plt.plot(gyr_osim[:, 0], 'r--', label='x sim imu')
    plt.plot(gyr_osim[:, 1], 'g--', label='y sim imu')
    plt.plot(gyr_osim[:, 2], 'b--', label='z sim imu')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title(imu_name)
    plt.subplot(2, 1, 2)
    plt.plot(acc_imu[:, 0], 'r', label='x imu')
    plt.plot(acc_imu[:, 1], 'g', label='x imu')
    plt.plot(acc_imu[:, 2], 'b', label='x imu')
    plt.plot(acc_osim[:, 0], 'r--', label='x sim imu')
    plt.plot(acc_osim[:, 1], 'g--', label='y sim imu')
    plt.plot(acc_osim[:, 2], 'b--', label='z sim imu')
    plt.ylabel('Fee Acc (m/s2)')
    plt.show()
    pass


def plot_segmented_gyro(x, side):
    plt.figure()
    for s in range(int(len(side) / 2)):
        plt.subplot(len(side) / 2, 1, s + 1)
        if side[0] == 'L':
            plt.plot(x['LFootIMU'][2*s][:, 4], label='Foot L Sensor_' + side[2 * s])
            plt.plot(x['RFootIMU'][2*s+1][:, 4], label='Foot R Sensor_' + side[2 * s + 1])
        elif side[0] == 'R':
            plt.plot(x['LFootIMU'][2*s+1][:, 4], label='Foot L Sensor_' + side[2 * s + 1])
            plt.plot(x['RFootIMU'][2*s][:, 4], label='Foot R Sensor_' + side[2 * s])
        plt.legend()
    plt.show()


def plot_segmented_ik(x, side, ik_r, ik_l):
    plt.figure()
    for s in range(int(len(side) / 2)):
        plt.subplot(len(side) / 2, 1, s + 1)
        if side[0] == 'L':
            plt.plot(x[2 * s][ik_l].values, label=ik_l + '_' + side[2 * s])
            plt.plot(x[2 * s][ik_r].values, label=ik_r + '_' + side[2 * s])
            plt.plot(x[2 * s + 1][ik_l].values, label=ik_l + '_' + side[2 * s + 1])
            plt.plot(x[2 * s + 1][ik_r].values, label=ik_r + '_' + side[2 * s + 1])

        elif side[0] == 'R':
            plt.plot(x[2 * s + 1][ik_r].values, label=ik_r + '_' + side[2 * s + 1])
            plt.plot(x[2 * s + 1][ik_l].values, label=ik_l + '_' + side[2 * s + 1])
            plt.plot(x[2 * s][ik_l].values, label=ik_l + '_' + side[2 * s])
            plt.plot(x[2 * s][ik_r].values, label=ik_r + '_' + side[2 * s])

        plt.legend()
    plt.show()


def plot_sdtw(long_seq, short_seq, mat, paths):
    plt.figure()
    sz1 = len(long_seq)
    sz2 = len(short_seq)
    n_repeat = 3
    # definitions for the axes
    left, bottom = 0.01, 0.1
    h_ts = 0.2
    w_ts = h_ts / n_repeat
    left_h = left + w_ts + 0.02
    width = height = 0.65
    bottom_h = bottom + height + 0.02

    rect_s_y = [left, bottom, w_ts, height]
    rect_gram = [left_h, bottom, width, height]
    rect_s_x = [left_h, bottom_h, width, h_ts]

    ax_gram = plt.axes(rect_gram)
    ax_s_x = plt.axes(rect_s_x)
    ax_s_y = plt.axes(rect_s_y)

    ax_gram.imshow(np.sqrt(mat))
    ax_gram.axis("off")
    ax_gram.autoscale(False)

    # Plot the paths
    for path in paths:
        ax_gram.plot([j for (i, j) in path], [i for (i, j) in path], "-",
                     linewidth=3.)

    ax_s_x.plot(np.arange(sz1), long_seq, "b-", linewidth=3.)
    ax_s_x.axis("off")
    ax_s_x.set_xlim((0, sz1 - 1))

    ax_s_y.plot(- short_seq, np.arange(sz2)[::-1], "b-", linewidth=3.)
    ax_s_y.axis("off")
    ax_s_y.set_ylim((0, sz2 - 1))
    plt.show()


def boxplot_pca_train_test(train_x_pca, test_x_pca, train_y_pca, test_y_pca, n_pca=5):
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.boxplot(train_x_pca[:, 0:n_pca])
    plt.title('train_x')
    plt.subplot(2, 2, 2)
    plt.boxplot(test_x_pca[:, 0:n_pca])
    plt.title('test_x')
    plt.subplot(2, 2, 3)
    plt.boxplot(train_y_pca[:, 0:n_pca])
    plt.title('train_y')
    plt.subplot(2, 2, 4)
    plt.boxplot(test_y_pca[:, 0:n_pca])
    plt.title('test_y')
    plt.show()


def scatter_pca_train_test(train_pca, test_pca, train_labels, test_labels, status):
    colorsb = plt.cm.tab20b((4. / 3 * np.arange(20 * 3 / 4)).astype(int))
    colorsc = plt.cm.tab20c((4. / 3 * np.arange(20 * 3 / 4)).astype(int))
    colors = np.concatenate([colorsb, colorsc])
    colors = ['r', 'g', 'c']
    test_subjects = test_labels['subjects'].unique()
    plt.figure()
    plt.scatter(x=train_pca[:, 0], y=train_pca[:, 1], color='b', label='train')
    for s, test_subject in enumerate(test_subjects):
        index = np.where(test_labels['subjects'] == test_subject)[0]
        c = colors[s]
        plt.scatter(x=test_pca[index, 0], y=test_pca[index, 1], color=c, label=test_subject)
    plt.title(status)
    plt.legend()
    plt.show()


def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def plot_gmm(gmm, X, label=False, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


def line_2d_original_new(x, x_new):
    if len(x.shape)==1:
        plt.figure()
        plt.subplot(3, 1, 1)
        for i in range(x.shape[0]):
            plt.plot(x[i, :], c='b')
            plt.title('original')
        plt.subplot(3, 1, 2)
        for i in range(x_new.shape[0]):
            plt.plot(x_new[i, :], c='g')
            plt.title('new')
        plt.subplot(3, 1, 3)
        for i in range(x.shape[0]):
            plt.plot(x[i, :], c='b')
        for i in range(x_new.shape[0]):
            plt.plot(x_new[i, :], c='g')
            plt.legend('original', 'new')
            plt.title('original new')
        plt.show()
    else:
        for c in range(x.shape[2]):
            plt.figure()
            plt.subplot(3, 1, 1)
            for i in range(x.shape[0]):
                plt.plot(x[i, :, c], c='b')
                plt.title('original')
            plt.subplot(3, 1, 2)
            for i in range(x_new.shape[0]):
                plt.plot(x_new[i, :, c], c='g')
                plt.title('new')
            plt.subplot(3, 1, 3)
            for i in range(x.shape[0]):
                plt.plot(x[i, :, c], c='b')
            for i in range(x_new.shape[0]):
                plt.plot(x_new[i, :, c], c='g')
                plt.legend('original', 'new')
                plt.title('original new')
        plt.show()


def scatter_2d_original_new(x_2d_original, x_2d_new, yhat, yhat_new):
    colors = ['b', 'g', 'r', 'c', 'k', 'y', 'm', 'lime', 'gold', 'salmon']
    colors = [plt.cm.tab20(i) for i in range(20)]
    colors.extend([plt.cm.tab20b(i) for i in range(20)])
    colors = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]
    plt.figure()
    i = 0
    for cluster in np.unique(yhat):
        plt.subplot(1, 2, 1)
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(x_2d_original[row_ix, 0], x_2d_original[row_ix, 1], label=cluster,
                    marker='o', c=colors[i], edgecolors=colors[i], alpha=0.5)
        plt.title('original')
        plt.subplot(1, 2, 2)
        plt.title('original + New')
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(x_2d_original[row_ix, 0], x_2d_original[row_ix, 1], label=cluster,
        marker='o', c=colors[i], edgecolors=colors[i], alpha=0.5)
        try:
            # get row indexes for samples with this cluster
            row_ix = np.where(yhat_new == cluster)
            # create scatter of these samples
            plt.scatter(x_2d_new[row_ix, 0], x_2d_new[row_ix, 1],
                        marker='^', c=colors[i], edgecolors=colors[i], alpha=0.5)
        except:
            continue
        i = i + 1
    plt.legend()
    plt.show()


def multicolor_pc_line(pca_comp_abs, kinematic_mean, plt_title):
    pc_indx = np.argmax(pca_comp_abs, axis=0)
    colors = [plt.cm.tab20(i) for i in range(20)]
    colors = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]
    cmap = ListedColormap(colors[0:pca_comp_abs.shape[0]])
    # cmap = 'hot'
    x = np.arange(0, len(kinematic_mean))
    y = kinematic_mean
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, pca_comp_abs.shape[0])
    lc = LineCollection(segments, cmap=cmap, linewidths=4, norm=norm)
    fig, ax = plt.subplots(2, 1)
    ax[0].set_xlim(np.min(x), np.max(x))
    ax[0].set_ylim(np.min(y), np.max(y))
    lc.set_array(pc_indx)
    line = ax[0].add_collection(lc)
    cbar = fig.colorbar(line, ax=ax[0])
    cbar.set_ticks(list(np.arange(0 + 1, pca_comp_abs.shape[0] + 1)))
    cbar.set_ticklabels(["PC " + str(i) for i in list(np.arange(0 + 1, pca_comp_abs.shape[0] + 1))])
    # plt.sci(lc)  # This allows interactive changing of the colormap.
    ax[0].set_title(plt_title)
    for i in range(len(pca_comp_abs)):
        ax[1].plot(pca_comp_abs[i, :], label='PC{}'.format(i + 1), linestyle='-',
                   c=colors[i])

    ax[1].set_xlim(np.min(x), np.max(x))
    cbar1 = fig.colorbar(line, ax=ax[1])
    cbar1.set_ticks(list(np.arange(0 + 1, pca_comp_abs.shape[0] + 1)))
    cbar1.set_ticklabels(["PC " + str(i) for i in list(np.arange(0 + 1, pca_comp_abs.shape[0] + 1))])
    # ax[1].legend()
    plt.show()


def plot_kinematics(data, dof_labels, activity, range_len, wandb_plot=True):
    plt.figure(figsize=(10, 8))
    n_kinematic = len(dof_labels)
    c = 2
    for i, kinematic_label in enumerate(dof_labels):
        plt.subplot(round(n_kinematic / c), c, i + 1)
        len_data = range(i * range_len, i * range_len + range_len)
        for j in data:
            plt.plot(np.arange(0, len(len_data)), j[len_data])
        if i == 1:
            plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.ylabel(kinematic_label + ' (deg)')
    plt.suptitle(activity, fontsize=16)
    plt.subplots_adjust(right=0.8)
    if wandb_plot:
        wandb.log({"kinematics": [wandb.Image(plt, caption=activity)]})
    else:
        plt.show()

def plot_kinematics_mean_std(data, knee_index, dof_labels, activity, range_len, wandb_plot=True):
    # plt.figure()
    plt.figure(figsize=(10, 8))
    n_kinematic = len(dof_labels)
    c = 2
    colors = ['b', 'r']
    for i, kinematic_label in enumerate(dof_labels):
        plt.subplot(round(n_kinematic / c), c, i + 1)
        len_data = range(i * range_len, i * range_len + range_len)
        for k, (key, value) in enumerate(knee_index.items()):
            # plt.fill_between(np.arange(0, len(len_data)),
            #                  np.mean(data[value, :], 0)[len_data] - np.std(data[value, :], 0)[len_data],
            #                  np.mean(data[value, :], 0)[len_data] + np.std(data[value, :], 0)[len_data],
            #                  label='+-std', color=colors[k], alpha=0.1)
            for j in range(len(value)):
                plt.plot(np.arange(0, len(len_data)), data[j, :][len_data], color='gray', alpha=0.1)
            # plt.plot(np.arange(0, len(len_data)), np.percentile(data[value, :], 5, axis=0)[len_data],
            #          label='5p_' + key, color=colors[k], alpha=0.9, linestyle='dashed')
            # plt.plot(np.arange(0, len(len_data)), np.percentile(data[value, :], 95, axis=0)[len_data],
            #          label='95p_' + key, color=colors[k], alpha=0.9, linestyle='dotted')
            # plt.plot(np.arange(0, len(len_data)),
            #          np.mean(data[value, :], 0)[len_data] - 2*np.std(data[value, :], 0)[len_data],
            #          label='-2std_' + key, color=colors[k], alpha=0.9, linestyle='dashed')
            # plt.plot(np.arange(0, len(len_data)),
            #          np.mean(data[value, :], 0)[len_data] + 2*np.std(data[value, :], 0)[len_data],
            #          label='+2std_' + key, color=colors[k], alpha=0.9, linestyle='dotted')

            plt.plot(np.mean(data[value, :], 0)[len_data], label='mean_' + key, c=colors[k])
        if i == 1:
            plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.ylabel(kinematic_label + ' (deg)')
    plt.suptitle(activity, fontsize=16)
    plt.subplots_adjust(right=0.8)
    if wandb_plot:
        wandb.log({"kinematics_mean_std": [wandb.Image(plt, caption=activity)]})
    else:
        plt.show()


def plot_pcs_variance_ratio(pca_variance, pca_variance_ratio, title, wandb_plot):
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(1, len(pca_variance) + 1), pca_variance, label='pca_variance')
    plt.ylabel('variance')
    plt.xlabel('PCs')

    plt.subplot(2, 1, 2)
    plt.bar(np.arange(1, len(pca_variance_ratio) + 1), pca_variance_ratio, label='pca_variance_ratio')
    plt.plot(np.cumsum(pca_variance_ratio), label='Cumulative variance (%)', c='k')
    plt.ylabel('variance ratio')
    plt.xlabel('PCs')
    plt.suptitle(title, fontsize=16)
    if wandb_plot:
        wandb.log({"pcs_variance_ratio": [wandb.Image(plt, caption=title)]})
    else:
        plt.show()


def plot_pcs_mode_profile(data, dof_labels, activity, mode_variation, range_len, n_mode, wandb_plot):
    if n_mode == None or n_mode <= 4:
        colors = ['b', 'r', 'g', 'c']
    else:
        colors = [plt.cm.tab20(i) for i in range(20)]
        colors = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]
    colors = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]
        # n = plt.cm.get_cmap("jet").N
        # colors = [plt.cm.jet(i) for i in range(0, n, round(n/n_mode))]
    mode_variation = {key: value for key, value in mode_variation.items() if 'pc4' in key or 'pc4' in key}
    colors = colors[3:]
    plt.figure(figsize=(10, 8))
    n_kinematic = len(dof_labels)
    c = 2
    for i, kinematic_label in enumerate(dof_labels):
        plt.subplot(round(n_kinematic / c), c, i + 1)
        len_data = range(i * range_len, i * range_len + range_len)
        plt.fill_between(np.arange(0, len(len_data)),
                         np.mean(data, 0)[len_data] - np.std(data, 0)[len_data],
                         np.mean(data, 0)[len_data] + np.std(data, 0)[len_data],
                         label='+-std', color='k', alpha=0.1)
        plt.plot(np.mean(data, 0)[len_data], label='mean', c='k')
        j = 0
        for m, (key, value) in enumerate(mode_variation.items()):
            if j % 2 == 0:
                linestyle = 'solid'
            else:
                linestyle = 'dashed'
            j +=1
            plt.plot(value[len_data], label=key, c=colors[math.floor(m/2)], linestyle=linestyle)
        if i == 1:
            plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.ylabel(kinematic_label + ' (deg)')
    plt.suptitle(activity, fontsize=16)
    plt.subplots_adjust(right=0.8)
    if wandb_plot:
        wandb.log({"pcs_mode_profile": [wandb.Image(plt, caption=activity)]})
    else:
        plt.show()



def plot_pcs_mode_profile_by_group(data, dof_labels, activity, mode_variation_by_group, range_len, n_mode, group_index, colors_groups, wandb_plot):

    if n_mode == None or n_mode <= 3:
        colors = ['b', 'r', 'g', 'c']
    else:
        colors = [plt.cm.tab20(i) for i in range(20)]
    colors = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]
    # colors_groups = ['b', 'r']
    n_kinematic = len(dof_labels)
    # reverse the nested mode_variation_by_group dictionary
    flipped_mode_variation_by_group = defaultdict(dict)
    for key, val in mode_variation_by_group.items():
        for subkey, subval in val.items():
            flipped_mode_variation_by_group[subkey][key] = subval
    c = 2
    for m, (key, value) in enumerate(flipped_mode_variation_by_group.items()):
        if m % 2 == 0:
            plt.figure(figsize=[10,8])
            linestyle = 'dotted'
        else:
            linestyle = 'dashed'

        for i, kinematic_label in enumerate(dof_labels):
            plt.subplot(round(n_kinematic / c), c, i + 1)
            len_data = range(i * range_len, i * range_len + range_len)
            for g, (key_g, value_g) in enumerate(value.items()):
                # mean and std groups
                # plt.fill_between(np.arange(0, len(len_data)),
                #                  np.mean(data[group_index[key_g], :], 0)[len_data] -
                #                  np.std(data[group_index[key_g], :], 0)[
                #                      len_data],
                #                  np.mean(data[group_index[key_g], :], 0)[len_data] +
                #                  np.std(data[group_index[key_g], :], 0)[
                #                      len_data],
                #                  label='+-std_' + key_g[:-6], color=colors_groups[g], alpha=0.1)
                # plt.plot(np.arange(0, len(len_data)),
                #          np.mean(data[group_index[key_g], :], 0)[len_data] - np.std(data[group_index[key_g], :], 0)[
                #              len_data],
                #          label='-std_' + key_g[:-6], color=colors_groups[g], alpha=0.2)
                # plt.plot(np.arange(0, len(len_data)),
                #          np.mean(data[group_index[key_g], :], 0)[len_data] + np.std(data[group_index[key_g], :], 0)[
                #              len_data],
                #          label='+-std_' + key_g[:-6], color=colors_groups[g], alpha=0.2)
                plt.plot(np.mean(data[group_index[key_g], :], 0)[len_data], label='mean_' + key_g[:-6],
                         c=colors_groups[g],
                         linestyle='solid')
                # mode
                plt.plot(value_g[len_data], label=key_g[:-6] + '_' + key, c=colors_groups[g], linestyle=linestyle)
            if i == 1:
                plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
            plt.ylabel(kinematic_label + ' (deg)')
        plt.suptitle(activity + '_mode ' + key[-1], fontsize=16, color=colors[math.floor(m/2)])
        plt.subplots_adjust(right=0.8)
        if m % 2 == 1:
            if wandb_plot:
                wandb.log({"pcs_mode_profile_by_group_mode"+ key[-1]: [wandb.Image(plt, caption=activity + '_mode ' + key[-1])]})
            else:
                plt.show()


def plot_pcs_comp_profile(dof_labels, activity, pca_comp, range_len, abs_pcs=False, wandb_plot=True):
    if abs_pcs == True:
        pca_comp = abs(pca_comp)
        pca_title_label = '_abs_'
    else:
        pca_title_label = '_'
    plt.figure(figsize=[10,8])
    colors = [plt.cm.tab20(i) for i in range(20)]
    colors = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]
    n_kinematic = len(dof_labels)
    c = 2
    for i, kinematic_label in enumerate(dof_labels):
        plt.subplot(round(n_kinematic / c), c, i + 1)
        len_data = range(i * range_len, i * range_len + range_len)
        plt.hlines(0, range_len, 0, linestyles='-.', colors='k')
        for l in range(len(pca_comp)):
            plt.plot(pca_comp[l, len_data], label='PC{}'.format(l + 1), linestyle='-',
                     c=colors[l])
        plt.ylabel(kinematic_label)
    plt.legend()
    plt.suptitle(activity + pca_title_label + 'PC component score', fontsize=16)
    if wandb_plot:
        wandb.log({"pcs_comp/loading_profile": [wandb.Image(plt, caption=activity)]})
    else:
        plt.show()


def plot_pcs_segment_profile(data, dof_labels, activity, pca_comp_abs, range_len, n_pc_segment=None, wandb_plot=True):
    n_kinematic = len(dof_labels)
    c = 2
    fig, axes = plt.subplots(round(n_kinematic / c), c)
    for i, kinematic_label in enumerate(dof_labels):
        len_data = np.arange(i * range_len, i * range_len + range_len)
        kinematic_mean = np.mean(data, 0)[len_data]
        if n_pc_segment == None or n_pc_segment>3:
            pca_comp_abs_seg = pca_comp_abs[:, :][:, len_data]
            if n_pc_segment:
                pca_comp_abs_seg = pca_comp_abs[0:n_pc_segment, :][:, len_data]
            colors = [plt.cm.tab20(i) for i in range(20)]
            colors = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]
        else:
            pca_comp_abs_seg = pca_comp_abs[0:n_pc_segment, :][:, len_data]
            colors = ['b', 'r', 'g', 'c']

        pc_indx = np.argmax(pca_comp_abs_seg, axis=0)
        cmap = ListedColormap(colors[0:pca_comp_abs_seg.shape[0]])
        # cmap = 'hot'
        x = np.arange(0, len(kinematic_mean))
        y = kinematic_mean
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, pca_comp_abs_seg.shape[0])
        lc = LineCollection(segments, cmap=cmap, linewidths=4, norm=norm)
        axes[i // 2, i % 2].set_xlim(np.min(x), np.max(x))
        axes[i // 2, i % 2].set_ylim(np.min(y), np.max(y))
        lc.set_array(pc_indx)
        line = axes[i // 2, i % 2].add_collection(lc)
        cbar = fig.colorbar(line, ax=axes[i // 2, i % 2])
        cbar.set_ticks(list(np.arange(0 + 1, pca_comp_abs_seg.shape[0] + 1)))
        cbar.set_ticklabels(["PC " + str(i) for i in list(np.arange(0 + 1, pca_comp_abs_seg.shape[0] + 1))])
        # plt.sci(lc)  # This allows interactive changing of the colormap.
        axes[i // 2, i % 2].set_title(kinematic_label)
    fig.suptitle(activity, fontsize=16)
    if wandb_plot:
        wandb.log({"pcs_segment_profile": [wandb.Image(fig, caption=activity)]})
    else:
        plt.show()


def plot_range_of_each_pc_mode(dof_labels, activity, mode_variation_df, range_len, wandb_plot):
    n_kinematic = len(dof_labels)
    # update the columns names
    new_columns_names = []
    for n in range(int(mode_variation_df.columns.size / 2)):
        new_columns_names.extend(['+{}std_PC{}'.format(2, n + 1)])
        new_columns_names.extend(['-{}std_PC{}'.format(2, n + 1)])
    mode_variation_df = pd.DataFrame(mode_variation_df.values, columns=new_columns_names)

    # set the plot color
    colors = [2 * [plt.cm.tab10(i)] for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]
    colors = sns.color_palette("tab10", 20)
    colors_palette = colors.copy()
    for i in range(20):
        if i % 2 == 0:
            c = colors[math.floor(i / 2)]
            c = tuple([i * 0.9 for i in c])
        else:
            c = colors[math.floor(i / 2)]
        colors_palette[i] = c

    c = 2
    fig, axes = plt.subplots(round(n_kinematic / c), c, figsize=[10, 8])
    for i, kinematic_label in enumerate(dof_labels):
        len_data = range(i * range_len, i * range_len + range_len)
        # mode_variation_df_temp = mode_variation_df.iloc[len_data].agg(['min', 'max'])
        mode_variation_df_temp = mode_variation_df.iloc[len_data]
        sns.boxplot(ax=axes[i // 2, i % 2], x="variable", y="value", data=mode_variation_df_temp.melt(),
                    palette=colors_palette)
        axes[i // 2, i % 2].set_ylabel(kinematic_label)
        axes[i // 2, i % 2].tick_params(axis='x', rotation=45)
        if i not in [6, 7]:
            axes[i // 2, i % 2].set_xticks([])
            axes[i // 2, i % 2].set(xlabel=None)

        if i!=0:
            axes[i // 2, i % 2].legend().set_visible(False)
    fig.suptitle(activity)
    if wandb_plot:
        wandb.log({"range_of_each_pc_mode": [wandb.Image(fig, caption=activity)]})
    else:
        plt.show()


def plot_range_of_each_pc_mode_by_group(dof_labels, activity, mode_variation_by_group, range_len, group_name, group_index, palette, wandb_plot):
    # set the plot color
    mode_variation_by_group_all = mode_variation_by_group.copy()
    for g, (key, value) in enumerate(mode_variation_by_group_all.items()):
        mode_variation_by_group_all[key] = pd.DataFrame.from_dict(value)
    mode_variation_by_group_all = pd.concat(mode_variation_by_group_all)
    # update the columns names
    new_columns_names = []
    for n in range(int(mode_variation_by_group_all.columns.size / 2)):
        new_columns_names.extend(['+{}std_PC{}'.format(2, n + 1)])
        new_columns_names.extend(['-{}std_PC{}'.format(2, n + 1)])
    mode_variation_by_group_all = pd.DataFrame(mode_variation_by_group_all.values, columns=new_columns_names,
                                               index=mode_variation_by_group_all.index.values)
    ## TO DO: should be fixed (given group name, and index, and color for group bar)
    # add knee columns
    k = [mode_variation_by_group_all.index.values[s][0] for s in range(len(mode_variation_by_group_all))]
    # group_name = 'knee'
    mode_variation_by_group_all[group_name] = k
    mode_variation_by_group_all[group_name] = mode_variation_by_group_all[group_name].astype('category')
    mode_variation_by_group_all = mode_variation_by_group_all.reset_index(drop=True)

    mode_variation_by_group_all_dic_df = {}
    for key in group_index:
        mode_variation_by_group_all_dic_df[key] = mode_variation_by_group_all[mode_variation_by_group_all[group_name] == key]
    c = 2
    n_kinematic = len(dof_labels)
    fig, axes = plt.subplots(round(n_kinematic / c), c, figsize=[10, 8])
    for i, kinematic_label in enumerate(dof_labels):
        len_data = range(i * range_len, i * range_len + range_len)

        mode_variation_df_temp_groups = []
        for key, value in mode_variation_by_group_all_dic_df.items():
            # mode_variation_df_temp = value.iloc[len_data].agg(['min', 'max'])
            mode_variation_df_temp = value.iloc[len_data]
            mode_variation_df_temp[group_name] = key[:-6]
            mode_variation_df_temp_groups.append(mode_variation_df_temp)
        mode_variation_df_temp = pd.concat(mode_variation_df_temp_groups).reset_index(
            drop=True)
        sns.boxplot(ax=axes[i // 2, i % 2], x="variable", y="value", data=mode_variation_df_temp.melt(id_vars=[group_name]),
                    hue=group_name, palette=palette)
        axes[i // 2, i % 2].set_ylabel(kinematic_label)
        axes[i // 2, i % 2].tick_params(axis='x', rotation=45)
        if i not in [6, 7]:
            axes[i // 2, i % 2].set_xticks([])
            axes[i // 2, i % 2].set(xlabel=None)

        if i!=0:
            axes[i // 2, i % 2].legend().set_visible(False)
    fig.suptitle(activity)
    if wandb_plot:
        wandb.log({"range_of_each_pc_mode_by_group": [wandb.Image(fig, caption=activity)]})
    else:
        plt.show()


def plot_heatmap(data_df, xlabel='', ylabel='', title='', subtitle='', wandb_plot=True):
    fig, axes = plt.subplots(figsize=[10, 8])
    f = sns.heatmap(data=data_df, cmap="coolwarm", annot=True, fmt='.3g', ax=axes)
    f.set_xlabel(xlabel)
    f.set_ylabel(ylabel)
    fig.tight_layout()
    if wandb_plot:
        wandb.log({"heatmap:" + title: wandb.Image(f)})
    else:
        plt.title('heatmap: ' + title + '|' + subtitle)
        plt.show()


def plot_warping_kiha(s1, s2, dof_labels, range_len, title, all_dof_together, filename=None):
    """Plot the optimal warping between to sequences.

    :param s1: From sequence.
    :param s2: To sequence.
    :param path: Optimal warping path.
    :param filename: Filename path (optional).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.patches import ConnectionPatch
    except ImportError:
        logger.error("The plot_warp function requires the matplotlib package to be installed.")
        return
    ''''''''''''''''''''''''
    if all_dof_together:
        # dtw_distance = dtw.distance(s1, s2)
        dtw_distance = dtw.distance_fast(s1, s2, use_pruning=True)
        ed_distance = ed.distance(s1, s2)
        path = dtw.warping_path(s1, s2)
        corr_r = scipy.stats.pearsonr(s1, s2)[0]
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
        ax[0].plot(s1)
        ax[1].plot(s2)
        plt.tight_layout()
        lines = []
        line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
        for r_c, c_c in path:
            if r_c < 0 or c_c < 0:
                continue
            con = ConnectionPatch(xyA=[r_c, s1[r_c]], coordsA=ax[0].transData,
                                  xyB=[c_c, s2[c_c]], coordsB=ax[1].transData, **line_options)
            lines.append(con)
        for line in lines:
            fig.add_artist(line)

        # t = ax[0].text(0.5, 0.5, 'distance measure=%f' % distance)
        # t.set_ha('center')
        plt.suptitle(title + '\n' + 'dtw distance=%f' % dtw_distance +
                     '\n' + 'euclidean  distance=%f' % ed_distance +
                     '\n' + 'r correlation =%f' % corr_r)
    else:
        fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(10, 10))
        for i, kinematic_label in enumerate(dof_labels):
            len_data = range(i * range_len, i * range_len + range_len)
            # dtw_distance = dtw.distance(s1[len_data], s2[len_data])
            dtw_distance = dtw.distance_fast(s1[len_data], s2[len_data], use_pruning=True)
            path = dtw.warping_path(s1[len_data], s2[len_data])
            ed_distance = ed.distance(s1[len_data], s2[len_data])
            corr_r = scipy.stats.pearsonr(s1[len_data], s2[len_data])[0]
            if i % 2 == 0:
                ax0 = axes[i, 0]
                ax1 = axes[i + 1, 0]
            else:
                ax0 = axes[i - 1, 1]
                ax1 = axes[i, 1]

            ax0.plot(s1[len_data])
            ax1.plot(s2[len_data])
            lines = []
            line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
            for r_c, c_c in path:
                if r_c < 0 or c_c < 0:
                    continue
                con = ConnectionPatch(xyA=[r_c, s1[len_data][r_c]], coordsA=ax0.transData,
                                      xyB=[c_c, s2[len_data][c_c]], coordsB=ax1.transData, **line_options)
                lines.append(con)
            for line in lines:
                fig.add_artist(line)
            # t = ax0.text(1, 1, 'distance measure=%f' % distance)
            # t.set_ha('center')
            ax0.set_title('dtw distance=%f' % dtw_distance +
                     '\n' + 'euclidean  distance=%f' % ed_distance +
                     '\n' + 'r correlation =%f' % corr_r)
            plt.suptitle(title)
            del ax0, ax1
    # if filename:
    #     plt.savefig(filename)
    #     plt.close()
    #     fig, ax = None, None
    # return fig, axes


def plot_cluster_dtw(timeseries_df, cluster_method, dof_labels, range_len, title, all_dof_together, filename=None):
    if all_dof_together:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        if cluster_method == 'linkage':
            model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
        else:
            model = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
        timeseries = timeseries_df.values.T
        cluster_idx = model.fit(timeseries)
        show_ts_label = lambda idx: "ts-" + str(idx)
        model.plot(axes=ax, show_ts_label=show_ts_label,
                   show_tr_label=True, ts_label_margin=-10,
                   ts_left_margin=10, ts_sample_length=2)
        plt.tight_layout()

    else:
        figs, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
        for i, kinematic_label in enumerate(dof_labels):
            len_data = range(i * range_len, i * range_len + range_len)
            if cluster_method == 'linkage':
                model = clustering.LinkageTree(dtw.distance_matrix_fast, {})
            else:
                model = clustering.HierarchicalTree(dists_fun=dtw.distance_matrix_fast, dists_options={})
            timeseries = timeseries_df.values[len_data, :].T
            cluster_idx = model.fit(timeseries)
            show_ts_label = lambda idx: "ts-" + str(idx)
            if i%2==1:
                ax = axes[i // 2, 0:2]
            else:
                ax = axes[i // 2, 2:4]
            model.plot(axes=ax, show_ts_label=show_ts_label,
                        show_tr_label=True, ts_label_margin=-10,
                        ts_left_margin=10, ts_sample_length=1)
        plt.tight_layout()


def plot_cluster_heatmap(data_df, method='single', metric='euclidean',
                         xlabel='', ylabel='', title='', subtitle='', wandb_plot=True):
    fig, axes = plt.subplots(figsize=[10, 8])
    f = sns.clustermap(data_df, method=method, metric=metric, cmap="PuOr", annot=True, fmt='.2f')
    f.ax_heatmap.set_xlabel(xlabel)
    f.ax_heatmap.set_ylabel(ylabel)
    plt.tight_layout()
    if wandb_plot:
        wandb.log({"cluster_heatmap:" + title: [wandb.Image(f, caption=subtitle)]})
    else:
        plt.show()



def plot_box_plot_pcs_hue(data_df, x="value", y="variable", hue="knee", palette="Set1", orient='h',
                          xlabel='', ylabel='', title='', subtitle='', wandb_plot=True):
    fig, axes = plt.subplots(figsize=[10, 8])
    f = sns.boxplot(x=x, y=y, hue=hue,
                     data=data_df, palette=palette, orient=orient, ax=axes)
    f.set_xlabel(xlabel)
    f.set_ylabel(ylabel)
    # plt.title(title + subtitle)
    plt.tight_layout()
    if wandb_plot:
        wandb.log({"Box Plot:" + title: [wandb.Image(plt, caption=subtitle)]})
    else:
        plt.show()


def plot_graph_network(data_df1, data_df2=None, metric='correlation', corr_method='pearson', lower_bound_threshold=0.4, upper_bound_threshold=0.9, title='title', wandb_plot=True):
    plt.figure(figsize=[10, 8])
    if data_df2 is None:
        vertices = data_df1.columns.values.tolist()
        if metric == 'correlation':
            edges = [((u, v), data_df1[u].corr(data_df1[v], method=corr_method)) for u, v in itertools.combinations(vertices, 2)]
    else:
        edges = []
        vertices_data_df1 = data_df1.columns.values.tolist()
        vertices_data_df2 = data_df2.columns.values.tolist()
        for u in vertices_data_df1:
            for v in vertices_data_df2:
                edges.append(((u, v), data_df1[u].corr(data_df2[v])))

    edges = [(u, v, {'weight': c}) for (u, v), c in edges if upper_bound_threshold>c >= lower_bound_threshold or -upper_bound_threshold<c<=-lower_bound_threshold]
    G = networkx.Graph()
    G.add_edges_from(edges)
    weights = [i[2]['weight'] for i in edges]
    nodes = G.nodes()
    degree = G.degree()
    color = [degree[n] for n in nodes]
    # color = 'b'
    size = [100 * (degree[n] + 1.0) for n in nodes]
    pos = networkx.circular_layout(G)
    networkx.draw(G, pos=pos, nodelist=nodes, node_color=color, node_size=size, node_shape="s", alpha=0.7, with_labels=True, font_weight="bold", edge_color=weights, width=1.0, edge_cmap=plt.cm.PuOr)
    if wandb_plot:
        wandb.log({"Graph Network" + metric: [wandb.Image(plt, caption=title)]})
    else:
        plt.show()


def plot_bar_pc1s_pc2s(pca_comp2_df):
    pca_comp2_df['pc2s'] = pca_comp2_df.index
    pca_comp2_df2 = pca_comp2_df.reset_index(drop=True)
    pca_comp2_df2 = pca_comp2_df2.melt(id_vars=['pc2s'])
    sns.barplot(x='variable', y='value', hue='pc2s', data=pca_comp2_df2, palette="tab20")
    plt.xlabel('Input Variables of PC2 (patients + PCs calculated from previous step (Kinematics Variable--> PCs)')
    plt.ylabel('PCs Component Values')
    plt.tight_layout()


def plot_clustermap_pc1s_pc2s(pca_comp2_df, metric="correlation", wandb_plot=True):
    plt.figure(figsize=(10, 8))
    sns.clustermap(pca_comp2_df, metric=metric, cmap="coolwarm", annot=True, fmt='.1g')
    plt.tight_layout()


def plot_scatter_pair(data_df_pcs_subject_variables, hue_variable, title, wandb_plot):
    try:
        sns.pairplot(data_df_pcs_subject_variables, hue=hue_variable, corner=True, palette="Set2")
    except:
        sns.pairplot(data_df_pcs_subject_variables, hue=hue_variable, corner=True, palette="Set2", diag_kind="hist")
    plt.tight_layout()
    if wandb_plot:
        wandb.log({"scatter_pair:" + hue_variable: [wandb.Image(plt, caption=title)]})
    else:
        plt.show()