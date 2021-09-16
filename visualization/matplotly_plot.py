import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import math

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


def plot_pcs_mode_profile(data, dof_labels, activity, mode_variation, range_len, n_mode):
    if n_mode == None or n_mode <= 3:
        colors = ['b', 'r', 'g']
    else:
        colors = [plt.cm.tab20(i) for i in range(20)]

    plt.figure()
    n_kinematic = len(dof_labels)
    c = 2
    for i, kinematic_label in enumerate(dof_labels):
        plt.subplot(round(n_kinematic / c), c, i + 1)
        len_data = range(i * range_len, i * range_len + range_len)
        plt.fill_between(np.arange(0, len(len_data)),
                         np.mean(data, 0)[len_data] - np.std(data, 0)[len_data],
                         np.mean(data, 0)[len_data] + np.std(data, 0)[len_data],
                         label='+-std', color='k', alpha=0.2)
        plt.plot(np.mean(data, 0)[len_data], label='mean', c='k')
        j = 0
        for m, (key, value) in enumerate(mode_variation.items()):
            if j % 2 == 0:
                linestyle = 'solid'
            else:
                linestyle = 'dashed'
            j +=1
            plt.plot(value[len_data], label=key, c=colors[math.floor(m/2)], linestyle=linestyle)
        plt.ylabel(kinematic_label + ' (deg)')
    plt.suptitle(activity, fontsize=16)
    plt.legend()
    plt.show()


def plot_pcs_segment_profile(data, dof_labels, activity, pca_comp_abs, range_len, n_pc_segment=None):
    n_kinematic = len(dof_labels)
    c = 2
    fig, axes = plt.subplots(round(n_kinematic / c), c)
    for i, kinematic_label in enumerate(dof_labels):
        len_data = np.arange(i * range_len, i * range_len + range_len)
        kinematic_mean = np.mean(data, 0)[len_data]
        if n_pc_segment == None:
            pca_comp_abs_seg = pca_comp_abs[:, :][:, len_data]
            colors = [plt.cm.tab20(i) for i in range(20)]
        else:
            pca_comp_abs_seg = pca_comp_abs[0:n_pc_segment, :][:, len_data]
            colors = ['b', 'r', 'g', 'orange']

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