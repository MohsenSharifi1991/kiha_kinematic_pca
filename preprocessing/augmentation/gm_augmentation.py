'''
This class is for generating synthetic data from source domain distribution using mixture of gaussian.
This function draw a sample from distribution that:
a) has larger samples with lower probablity
b) has smaller sample in their distibution
c) or both a and b
The steps are as follow:
1) reshaping the 3d data to 2d. in: 3d kinematic ->f1 -> 2d kinematic
2) dimension reduction using i.g. PCA. in:2d kinematic(n*f) -> f2 -> 2d kinematic (n*c); c<f
3) form mixture of gaussian. in:2d kinematic (n*c) -> f3 -> model; P1, P2, P3, ... Pn
4) identify optimum component. find optimum n-->m
5) identify  distribution based on a, b, or c.  in: model;P1, P2, P3, ... Pm --> f5 -> p1, p4
6) draw sample from selected distribution or entire mixture of gaussian. P1, P4

functions: visualization
'''
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.mixture import GaussianMixture
from umap import UMAP
from smt.sampling_methods import LHS

from visualization.matplotly_plot import scatter_2d_original_new, line_2d_original_new
from visualization.wandb_plot import wandb_scatter_2d_train_test, \
    wandb_scatter_3d_train_test


class GMAugmentation:
    def __init__(self, x, embedding_method='PCA', embedding_n_component=5,
                 gm_sampling_method='smaller_dist', gm_n_sample=50, x_label=None, gm_n_component=None, plot_display=True):
        self.x = x
        self.x_axis1 = self.x.shape[1]
        self.x_axis2 = self.x.shape[2]
        self.embedding_method = embedding_method
        self.embedding_n_component = embedding_n_component
        self.gm_sampling_method = gm_sampling_method
        self.gm_n_sample = gm_n_sample
        self.gm_n_component = gm_n_component
        self.plot_display = plot_display
        self.x_label = x_label

    def run_embedding_visualization(self):
        self.reshape_3dto2d()
        self.apply_embedding()
        self.plot_embedding()

    def run_augmentation(self):
        self.reshape_3dto2d() # x3d->x2d
        self.apply_embedding() # x2d -> x2d (c)
        if self.embedding_method != 'LDA':
            self.get_gm_ncomponent() # x2d (c) -> gm_n_comp
            self.form_gm() # x2d (c), gm_n_comp -> gm
            self.draw_new_samples()  # gm, sampling_method, n_sample --> x2d_new (c)
            self.apply_inverse_embedding()  # x2d_new (c) -> x2d_new
        else:
            self.draw_new_samples()
        self.reshape_2dto3d() # x2d_new -> x3d_new
        if self.plot_display:
            # self.plot_original_new_line()
            self.plot_original_new_scatter()
        return self.x_new

    def reshape_3dto2d(self):
        self.x_2d = np.reshape(self.x,
                             [self.x.shape[0], self.x_axis1*self.x_axis2], 'F')

    def reshape_2dto3d(self):
        self.x_new = np.reshape(self.x_2d_new,
                             [self.x_2d_new.shape[0], self.x_axis1, self.x_axis2], 'F')

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


    def apply_embedding(self):
        if self.embedding_method =='PCA':
            self.x_2d_transformed, self.transformer_function = self.apply_pca(self.x_2d)
        elif self.embedding_method == 'Spectral':
            self.x_2d_transformed, self.transformer_function = self.apply_spectral(self.x_2d)
        elif self.embedding_method == 'tSNE':
            self.x_2d_transformed, self.transformer_function = self.apply_tsne(self.x_2d)
        elif self.embedding_method == 'UMAP':
            self.x_2d_transformed, self.transformer_function = self.apply_umap(self.x_2d)
        elif self.embedding_method == 'tSVD':
            self.x_2d_transformed, self.transformer_function = self.apply_tsvd(self.x_2d)
        elif self.embedding_method == 'LDA':
            self.x_2d_transformed, self.gm = self.apply_lda(self.x_2d, self.x_label)

    def apply_inverse_embedding(self):
        self.x_2d_new = self.transformer_function.inverse_transform(self.x_2d_transformed_new)

    def convert_np2df_labels(self, labels):
        if len(labels.shape)==2:
            return labels
        else:
            return pd.DataFrame(labels[:, 0, :], columns=self.config['label_headers'])

    def plot_original_new_line(self):
        line_2d_original_new(self.x, self.x_new)

    def plot_original_new_scatter(self):
        yhat = self.predict_yhat_gm(self.gm, self.x_2d_transformed)
        yhat_new = self.predict_yhat_gm(self.gm, self.x_2d_transformed_new)
        if self.x_label is not None:
            scatter_2d_original_new(self.x_2d_transformed, self.x_2d_transformed_new, self.x_label, yhat_new)
        scatter_2d_original_new(self.x_2d_transformed, self.x_2d_transformed_new, yhat, yhat_new)

    def plot_embedding(self,):
        if self.plot_method == '2d':
            wandb_scatter_2d_train_test(self.train_x_transformed, self.test_x_transformed, self.convert_np2df_labels(self.train_label),
                                        self.convert_np2df_labels(self.test_label), status_title=str('X_'+self.embedding_method+self.plot_method))
            wandb_scatter_2d_train_test(self.train_y_transformed, self.test_y_transformed, self.convert_np2df_labels(self.train_label),
                                        self.convert_np2df_labels(self.test_label), status_title=str('Y_'+self.embedding_method+self.plot_method))
        elif self.plot_method == '3d':
            wandb_scatter_3d_train_test(self.train_x_transformed, self.test_x_transformed, self.convert_np2df_labels(self.train_label),
                                        self.convert_np2df_labels(self.test_label), status_title=str('X_'+self.embedding_method+self.plot_method))
            wandb_scatter_3d_train_test(self.train_y_transformed, self.test_y_transformed, self.convert_np2df_labels(self.train_label),
                                        self.convert_np2df_labels(self.test_label), status_title=str('Y_'+self.embedding_method+self.plot_method))

    def apply_lda(self, x_2d, x_label):
        lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
        lda_fit = lda.fit(x_2d, x_label)
        x_transformed = lda_fit.transform(x_2d)
        return x_transformed, lda

    def apply_pca(self, x_2d):
        pca = PCA(n_components=self.embedding_n_component)
        x_transformed = pca.fit_transform(x_2d)
        return x_transformed, pca

    def apply_spectral(self, x_2d):
        spectral = SpectralEmbedding(n_neighbors=20, n_components=self.embedding_n_component)
        x_transformed = spectral.fit_transform(x_2d)
        return x_transformed, spectral

    def apply_tsne(self, x_2d):
        tsne = TSNE(n_components=self.embedding_n_component, perplexity=30, early_exaggeration=12, learning_rate=200.0, n_iter=1000)
        x_transformed = tsne.fit_transform(x_2d)
        return x_transformed, tsne

    def apply_umap(self, x_2d):
        umap = UMAP(n_components=self.embedding_n_component, init='random', random_state=0)
        x_transformed = umap.fit_transform(x_2d)
        return x_transformed, umap

    def apply_tsvd(self, x_2d):
        tsvd = TruncatedSVD(n_components=self.embedding_n_component, init='random', random_state=0, algorithm='randomized'),
        x_transformed = tsvd.fit_transform(x_2d)
        return x_transformed, tsvd

    def get_gm_ncomponent(self):
        if self.gm_n_component is None:
            n_components = np.arange(1, 26)
            gms = [GaussianMixture(n, covariance_type='full', random_state=0).fit(self.x_2d_transformed) for n in n_components]
            n_aic = np.argmin(np.asarray([m.aic(self.x_2d_transformed) for m in gms]))
            self.gm_n_component = n_aic
        return self.gm_n_component

    def form_gm(self):
        gm = GaussianMixture(n_components=self.gm_n_component, covariance_type='full', random_state=0)
        gm = gm.fit(self.x_2d_transformed)
        self.gm = gm

    def predict_yhat_gm(self, gm, x):
        return gm.predict(x)

    def predict_prob_gm(self, gm, x):
        return gm.predict_proba(x)

    def get_smaller_dist(self, gm, x):
        yhat = self.predict_yhat_gm(gm, x)
        min_idx = np.argmin(np.asarray([len(np.where(yhat==i)[0]) for i in range(self.gm_n_component)]))
        smaller_dist_component = min_idx
        return smaller_dist_component

    def get_lowprob_dist(self, gm, x):
        yprob = self.predict_prob_gm(gm, x)
        low_yprob = np.argwhere((0.1 < yprob) & (yprob < 0.9))[:, 1]
        min_idx = np.argmin(np.asarray([len(np.where(low_yprob==i)[0]) for i in range(self.gm_n_component)]))
        low_yprob_dist_component = min_idx
        return low_yprob_dist_component

    def draw_new_samples(self,):
        means = self.gm.means_
        covs = self.gm.covariances_
        n_sample = self.gm_n_sample
        if self.gm_sampling_method == 'general_dist':
            x_new = self.gm.sample(n_sample)[0]
        elif self.gm_sampling_method == 'smaller_dist':
            gaussian_index = self.get_smaller_dist(self.gm, self.x_2d_transformed)
            gcov = covs[gaussian_index].squeeze()
            gmean = means[gaussian_index].squeeze()
            x_new = np.random.multivariate_normal(gmean, gcov, n_sample)
        elif self.gm_sampling_method == 'lowproba_dist':
            gaussian_index = self.get_lowprob_dist(self.gm, self.x_2d_transformed)
            gcov = covs[gaussian_index].squeeze()
            gmean = means[gaussian_index].squeeze()
            x_new = np.random.multivariate_normal(gmean, gcov, n_sample)
        elif self.gm_sampling_method == 'lhs':
            xlim = self.x_2d_transformed[:, 0:2]
            sampling = LHS(xlimits=xlim, criterion='maximin')
            x_new = sampling(n_sample)

        self.x_2d_transformed_new = x_new





