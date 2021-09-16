from sklearn.decomposition import PCA
import numpy as np


class PCAnalysis:
    def __init__(self, x, pc_n_component=None, pc_variance= None):
        self.x = x
        self.x_axis1 = self.x.shape[1]
        self.x_axis2 = self.x.shape[2]
        self.pc_n_component = pc_n_component
        self.pc_variance = pc_variance

    def reshape_3dto2d(self):
        self.x_2d = np.reshape(self.x,
                               [self.x.shape[0], self.x_axis1 * self.x_axis2], 'F')

    def reshape_2dto3d(self):
        self.x_new = np.reshape(self.x_2d_new,
                                [self.x_2d_new.shape[0], self.x_axis1, self.x_axis2], 'F')

    def apply_pca(self, x_2d):
        if self.pc_n_component != None and self.pc_variance == None:
            pca = PCA(n_components=self.embedding_n_component)
        elif self.pc_n_component == None and self.pc_variance != None:
            pca = PCA(self.pc_variance)
        elif self.pc_n_component == None and self.pc_variance == None:
            pca = PCA()
        pca = pca.fit(x_2d)
        cov = pca.get_covariance()
        pca_comp = pca.components_                          # [n_component, n_features]  if don't chose any pca component or variance, then we will get n_component=n_sampel
        pca_comp_abs = abs(pca_comp)
        pca_mean = pca.mean_                                # [n_features,1] mean over features
        pca_variance = pca.explained_variance_              # [n_component,1] variance of each component
        pca_variance_ratio = pca.explained_variance_ratio_  # [n_component,1] Percentage of variance explained by each of the selected components.
        x_transformed = pca.fit_transform(x_2d)
        return pca, pca_comp, x_transformed

    def form_mode_variations(self, x_2d, n_mode, std_factor):
        pca, pca_comp, x_transformed = self.apply_pca(x_2d)
        # mean of PC scores
        mean_score = np.mean(x_transformed, 0)
        # Std of PC scores
        std_score = np.std(x_transformed, 0)
        mode_variation = {}
        for i in range(n_mode):
            mode_std = np.mean(x_2d, 0) + (mean_score[i] + std_factor * std_score[i]) * pca_comp[i, :]
            mode_nstd = np.mean(x_2d, 0) + (mean_score[i] - std_factor * std_score[i]) * pca_comp[i, :]
            mode_variation['mode + {}stdev_pc{}'.format(std_factor, i+1)] = mode_std
            mode_variation['mode - {}stdev_pc{}'.format(std_factor, i+1)] = mode_nstd
        return mode_variation

    def form_signal_patient_variables_data(self):
        return

    def apply_regression_on_signal_patient_variables(self):
        return

