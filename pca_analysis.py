'''
This class for 2d and 3d signals and corresponding patient demographic data
1) check if the signals is 2d or 3d. If 3d, then transform to 2d
2) conduct PC analysis on 2d signals
3) display the result of PCA from step 2

4) if patient demographic data exist, then form new data between transform 2d signals from step 2 and patient demographic variables
5) conduct analysis related to patient demographic variables and signals
6) statistical analysis: t test, anova, manova, or export data
7.a) method 1: pearson and spearman correlation
7.b) method 2: PCA on the form dataset
7.c) method 3: multivariate regression to go from signals to patient demographic variables and vise versa
'''
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from visualization.matplotly_plot import plot_pcs_variance_ratio, plot_pcs_mode_profile, plot_range_of_each_pc_mode, \
    plot_pcs_comp_profile, plot_pcs_segment_profile, plot_pcs_mode_profile_by_group, plot_range_of_each_pc_mode_by_group
from visualization.plot_4linkage_mechanism import Plot4LinkageMechanism


class PCAnalysis:
    def __init__(self, config, x, range_len=None, pc_n_component=None, pc_variance=None):
        self.config = config
        self.x = x
        self.range_len = range_len
        self.x_2d = []
        # check if the shape is 2d. If not, reshape data to 2d
        if self.range_len is None and len(self.x.shape)==2:
            self.x_axis1 = self.x.shape[1]
            self.x_axis2 = self.x.shape[2]
            self.range_len = self.x_axis1
            self.reshape_3dto2d()
        elif self.range_len is None and len(self.x.shape)==1:
            self.range_len = config['target_padding_length']
            self.x_2d = self.x
        else:
            self.range_len = range_len
            self.x_2d = self.x
        # self.demographic_variables = demographic_variables
        self.pc_n_component = pc_n_component
        self.pc_variance = pc_variance


    def reshape_3dto2d(self):
        self.x_2d = np.reshape(self.x,
                               [self.x.shape[0], self.x_axis1 * self.x_axis2], 'F')

    def reshape_2dto3d(self):
        self.x_new = np.reshape(self.x_2d_new,
                                [self.x_2d_new.shape[0], self.x_axis1, self.x_axis2], 'F')

    def apply_pca(self):
        if self.pc_n_component != None and self.pc_variance == None:
            pca = PCA(n_components=self.pc_n_component)
        elif self.pc_n_component == None and self.pc_variance != None:
            pca = PCA(self.pc_variance)
        elif self.pc_n_component == None and self.pc_variance == None:
            pca = PCA()
        self.pca = pca.fit(self.x_2d)
        cov = pca.get_covariance()
        self.pca_comp = pca.components_                          # [n_component, n_features]  if don't chose any pca component or variance, then we will get n_component=n_sampel, eigenvectors
        self.pca_comp_abs = abs(self.pca_comp)
        self.pca_loading_vector = pca.components_.T * np.sqrt(pca.explained_variance_)
        # self.pca_loading_vector = MinMaxScaler().fit_transform(self.pca_loading_vector)
        pca_mean = pca.mean_                                     # [n_features,1] mean over features
        self.pca_variance = pca.explained_variance_              # [n_component,1] variance of each component: The eigenvalues represent the variance in the direction of the eigenvector.
        self.pca_variance_ratio = pca.explained_variance_ratio_  # [n_component,1] Percentage of variance explained by each of the selected components.
        self.x_transformed = pca.fit_transform(self.x_2d)
        return self.pca, self.pca_comp, self.pca_loading_vector, self.pca_variance_ratio, self.x_transformed

    def display_pca_variance_ratio(self, title, wandb_plot):
        self.apply_pca()
        plot_pcs_variance_ratio(self.pca_variance, self.pca_variance_ratio, title, wandb_plot=wandb_plot)

    def form_mode_variations(self, n_mode, std_factor):
        self.apply_pca()
        # mean of PC scores
        mean_score = np.mean(self.x_transformed, 0)
        # Std of PC scores
        std_score = np.std(self.x_transformed, 0)
        mode_variation = {}
        for i in range(n_mode):
            mode_std = np.mean(self.x_2d, 0) + (mean_score[i] + std_factor * std_score[i]) * self.pca_comp[i, :]
            mode_nstd = np.mean(self.x_2d, 0) + (mean_score[i] - std_factor * std_score[i]) * self.pca_comp[i, :]
            mode_variation['mode + {}stdev_pc{}'.format(std_factor, i+1)] = mode_std
            mode_variation['mode - {}stdev_pc{}'.format(std_factor, i+1)] = mode_nstd
        self.mode_variation = mode_variation
        return self.mode_variation


    def form_mode_variations_by_group(self, n_mode, std_factor, group_index):
        self.apply_pca()
        self.mode_variation_by_group = {}
        for g, (key, value) in enumerate(group_index.items()):
            # mean of PC scores
            mean_score = np.mean(self.x_transformed[value, :], 0)
            # Std of PC scores
            std_score = np.std(self.x_transformed[value, :], 0)
            mode_variation = {}
            for i in range(n_mode):
                mode_std = np.mean(self.x_2d[value, :], 0) + (mean_score[i] + std_factor * std_score[i]) * self.pca_comp[i, :]
                mode_nstd = np.mean(self.x_2d[value, :], 0) + (mean_score[i] - std_factor * std_score[i]) * self.pca_comp[i, :]
                mode_variation['mode + {}stdev_pc{}'.format(std_factor, i+1)] = mode_std
                mode_variation['mode - {}stdev_pc{}'.format(std_factor, i+1)] = mode_nstd
            self.mode_variation_by_group[key] = mode_variation

        return self.mode_variation_by_group

    def display_pcs_mode_profile(self, n_mode, std_factor, title, selected_mode, wandb_plot):
        self.form_mode_variations(n_mode, std_factor)
        plot_pcs_mode_profile(self.x_2d, self.config['y_axis_lims'],self.config['selected_opensim_labels'], title + ':' + str(std_factor) + 'std',
                              self.mode_variation, self.range_len, n_mode=n_mode, selected_mode=selected_mode, wandb_plot=wandb_plot)

    def display_pcs_mode_profile_by_group(self, n_mode, std_factor, title, group_index, colors_groups, wandb_plot):
        self.form_mode_variations_by_group(n_mode, std_factor, group_index)
        plot_pcs_mode_profile_by_group(self.x_2d, self.config['selected_opensim_labels'], title + ':Mean-/+' + str(std_factor) + 'std',
                              self.mode_variation_by_group, self.range_len, n_mode=n_mode, group_index=group_index, colors_groups=colors_groups, wandb_plot=wandb_plot)

    def display_range_of_each_pc_mode(self, title, wandb_plot):
        plot_range_of_each_pc_mode(self.config['selected_opensim_labels'],  title,
                                             pd.DataFrame.from_dict(self.mode_variation), self.range_len, wandb_plot=wandb_plot)

    def display_range_of_each_pc_mode_by_group(self, title, group_name, group_index, wandb_plot, pallete):
        plot_range_of_each_pc_mode_by_group(self.config['selected_opensim_labels'], title,
                                             self.mode_variation_by_group, self.range_len, group_name, group_index, pallete, wandb_plot=wandb_plot)

    def display_pcs_comp_profile(self, title, abs_pcs_status=True, wandb_plot=True, selected_PCs=None):
        plot_pcs_comp_profile(self.config['selected_opensim_labels'], title,
                              self.pca_comp, self.range_len, abs_pcs=abs_pcs_status, wandb_plot=wandb_plot, selected_PCs=selected_PCs)

    def display_pcs_loading_profile(self, title, abs_pcs_status=True, wandb_plot=True, selected_PCs=['PC1', 'CP2']):
        plot_pcs_comp_profile(self.config['selected_opensim_labels'], title,
                              self.pca_loading_vector.T, self.range_len, abs_pcs=abs_pcs_status, wandb_plot=wandb_plot, selected_PCs=selected_PCs)

    def display_pcs_segment_profile(self, title, wandb_plot):
        # plot pcs segment
        plot_pcs_segment_profile(self.x_2d, self.config['y_axis_lims'], self.config['selected_opensim_labels'], title,
                                 abs(self.pca_loading_vector.T), self.range_len, n_pc_segment=None, wandb_plot=wandb_plot)


    def display_pcs_segment_profile_by_group(self, title, group_index, wandb_plot):
        # plot pcs segment this is not working since pca comp abs are done over both group
        for key, value in group_index.items():
            plot_pcs_segment_profile(self.x_2d, self.config['selected_opensim_labels'], title,
                                     self.pca_comp_abs, self.range_len, n_pc_segment=None, wandb_plot=wandb_plot)

    def display_4linkage_mechanism(self, n_mode, std_factor, title, wandb_plot):
        self.form_mode_variations(n_mode, std_factor)
        # plot 4 linkage mechanism
        linkage_mechanism_handler = Plot4LinkageMechanism(self.x_2d, self.config['selected_opensim_labels'], title + ':' + str(std_factor) + 'std',
                                                          self.mode_variation, self.range_len, wandb_plot=wandb_plot)
        linkage_mechanism_handler.plot_mechanism_row_basis(10, 60, 50, 20)

    def display_4linkage_mechanism_animation(self, n_mode, std_factor, title, wandb_plot):
        self.form_mode_variations(n_mode, std_factor)
        # plot 4 linkage mechanism
        linkage_mechanism_handler = Plot4LinkageMechanism(self.x_2d, self.config['selected_opensim_labels'], title + ':' + str(std_factor) + 'std',
                                                          self.mode_variation, self.range_len, wandb_plot=wandb_plot)
        linkage_mechanism_handler.animate_plot_mechanism(10, 60, 50, 20)