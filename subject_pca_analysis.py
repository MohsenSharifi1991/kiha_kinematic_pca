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
from sklearn.preprocessing import StandardScaler

from pca_analysis import PCAnalysis
from visualization.matplotly_plot import plot_pcs_variance_ratio, plot_pcs_mode_profile, plot_range_of_each_pc_mode, \
    plot_pcs_comp_profile, plot_pcs_segment_profile
from visualization.plot_4linkage_mechanism import Plot4LinkageMechanism


class SubjectPCsAnalysis:
    def __init__(self, config, subject_demographic_variables, x, range_len=None, pc_n_component=None, pc_variance=None,
                 pc_subject_n_component=None, pc_subject_variance=None, standardize_method=None):
        self.config = config
        self.subject_demographic_variables = subject_demographic_variables
        self.x = x
        self.range_len = range_len
        self.x_2d = []
        # self.demographic_variables = demographic_variables
        self.pc_n_component = pc_n_component
        self.pc_variance = pc_variance
        self.pc_subject_n_component = pc_subject_n_component
        self.pc_subject_variance = pc_subject_variance
        self.standardize_method = standardize_method
        pcanalysis_handler = PCAnalysis(config, self.x, 100, pc_variance=self.pc_variance)
        self.pca, self.pca_comp, self.pca_loading_vector, self.pca_variance_ratio, self.x_transformed = pcanalysis_handler.apply_pca()

    def standardize(self, standardize_method):
        # standrize the patient_pcs data
        if standardize_method == 'all':
            self.subject_pcs = self.form_subject_variables_signal_data()
            scaler = StandardScaler()
            scaler_fit = scaler.fit(self.subject_pcs)
            self.subject_pcs_scaled = scaler_fit.transform(self.subject_pcs)
        else:
            scaler = StandardScaler()
            scaler_fit = scaler.fit(self.subject_demographic_variables)
            subject_variables_scaled = scaler_fit.transform(self.subject_demographic_variables)
            scaler_pcs = StandardScaler()
            scaler_fit_pcs = scaler_pcs.fit(self.x_transformed.reshape(-1, 1))
            x_transformed_scaled = scaler_fit_pcs.transform(self.x_transformed.reshape(-1, 1))
            x_transformed_scaled = np.reshape(x_transformed_scaled,
                                                           [self.x_transformed.shape[0],
                                                            self.x_transformed.shape[1]], 'F')
            self.subject_pcs_scaled = np.concatenate([subject_variables_scaled, x_transformed_scaled], axis=1)
            return self.subject_pcs_scaled

    def apply_pca_subject(self):
        # run data standrize
        self.standardize(self.standardize_method)
        if self.pc_subject_n_component != None and self.pc_subject_variance == None:
            pca = PCA(n_components=self.pc_subject_n_component)
        elif self.pc_subject_n_component == None and self.pc_subject_variance != None:
            pca = PCA(self.pc_subject_variance)
        elif self.pc_subject_n_component == None and self.pc_subject_variance == None:
            pca = PCA()
        self.pca_subject = pca.fit(self.subject_pcs_scaled)
        cov = pca.get_covariance()
        self.pca_comp_subject = pca.components_                          # [n_component, n_features]  if don't chose any pca component or variance, then we will get n_component=n_sampel
        self.pca_comp_abs_subject = abs(self.pca_comp)
        pca_mean = pca.mean_                                             # [n_features,1] mean over features
        self.pca_variance_subject = pca.explained_variance_              # [n_component,1] variance of each component
        self.pca_variance_ratio_subject = pca.explained_variance_ratio_  # [n_component,1] Percentage of variance explained by each of the selected components.
        self.subject_pcs_scaled_transformed = pca.fit_transform(self.subject_pcs_scaled)
        return self.pca_subject, self.pca_comp_subject, self.pca_variance_ratio_subject, self.subject_pcs_scaled_transformed


    def display_pca_variance_ratio(self, title, wandb_plot=True):
        self.apply_pca_subject()
        plot_pcs_variance_ratio(self.pca_variance_subject, self.pca_variance_ratio_subject, title=title, wandb_plot=wandb_plot)

    def combine_PC1s_PC2s_df(self):
        pcs2_columns = ['pc2_' + str(i + 1) for i in range(self.subject_pcs_scaled_transformed.shape[1])]
        pcs1_columns = ['pc1_' + str(i + 1) for i in range(self.x_transformed.shape[1])]
        subject_demographic_headers = list(self.subject_demographic_variables.columns.values)
        subject_demographic_headers.extend(pcs1_columns)
        combined_pc1s_pc2s_df = pd.DataFrame(self.pca_comp_subject, columns=subject_demographic_headers, index=pcs2_columns)
        return combined_pc1s_pc2s_df

    def form_subject_variables_signal_data(self):
        self.subject_pcs = np.concatenate([self.subject_demographic_variables, self.x_transformed], axis=1)
        return self.subject_pcs

    def apply_regression_on_signal_subject_variables(self):
        return

