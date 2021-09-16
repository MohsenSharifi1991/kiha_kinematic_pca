import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.mixture import GaussianMixture
from umap import UMAP
from visualization.wandb_plot import wandb_scatter_2d_train_test, \
    wandb_scatter_3d_train_test


class EmbeddingVisualization:
    def __init__(self, config, kihadataset_train, kihadataset_test, n_component=5, embedding_method='PCA', plot_method='2d'):
        self.kihadataset_train = kihadataset_train
        self.kihadataset_test = kihadataset_test
        del kihadataset_train, kihadataset_test
        self.config = config
        self.train_x = self.kihadataset_train['x']
        self.train_y = self.kihadataset_train['y']
        self.train_label = self.kihadataset_train['labels']
        self.test_x = self.kihadataset_test['x']
        self.test_y = self.kihadataset_test['y']
        self.test_label = self.kihadataset_test['labels']
        self.n_component = n_component
        self.embedding_method = embedding_method
        self.plot_method = plot_method
        self.x_axis1 = self.train_x.shape[1]
        self.x_axis2 = self.train_x.shape[2]
        self.y_axis1 = self.train_y.shape[1]
        self.y_axis2 = self.train_y.shape[2]

    def run_embedding_visualization(self):
        self.reshape_3dto2d()
        self.apply_embedding()
        self.plot_embedding()

    def apply_embedding(self):
        if self.embedding_method =='PCA':
            self.train_x_transformed, self.test_x_transformed = self.apply_pca(self.train_x, self.test_x)
            self.train_y_transformed, self.test_y_transformed = self.apply_pca(self.train_y, self.test_y)
        elif self.embedding_method == 'Spectral':
            self.train_x_transformed, self.test_x_transformed = self.apply_spectral(self.train_x, self.test_x)
            self.train_y_transformed, self.test_y_transformed = self.apply_spectral(self.train_y, self.test_y)
        elif self.embedding_method == 'tSNE':
            self.train_x_transformed, self.test_x_transformed = self.apply_tsne(self.train_x, self.test_x)
            self.train_y_transformed, self.test_y_transformed = self.apply_tsne(self.train_y, self.test_y)
        elif self.embedding_method == 'UMAP':
            self.train_x_transformed, self.test_x_transformed = self.apply_umap(self.train_x, self.test_x)
            self.train_y_transformed, self.test_y_transformed = self.apply_umap(self.train_y, self.test_y)
        elif self.embedding_method == 'tSVD':
            self.train_x_transformed, self.test_x_transformed = self.apply_tsvd(self.train_x, self.test_x)
            self.train_y_transformed, self.test_y_transformed = self.apply_tsvd(self.train_y, self.test_y)

    def convert_np2df_labels(self, labels):
        if len(labels.shape)==2:
            return labels
        else:
            return pd.DataFrame(labels[:, 0, :], columns=self.config['label_headers'])

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

    def reshape_3dto2d(self):
        self.train_x = np.reshape(self.train_x,
                             [self.train_x.shape[0], self.x_axis1*self.x_axis2])
        self.train_y = np.reshape(self.train_y,
                             [self.train_y.shape[0], self.y_axis1*self.y_axis2])
        self.test_x = np.reshape(self.test_x,
                            [self.test_x.shape[0], self.x_axis1*self.x_axis2])
        self.test_y = np.reshape(self.test_y,
                            [self.test_y.shape[0], self.y_axis1*self.y_axis2])

    def apply_pca(self, train, test):
        pca = PCA(n_components=self.n_component)
        data_transformed = pca.fit_transform(np.concatenate([train, test], axis=0))
        train_transformed = data_transformed[0:train.shape[0]]
        test_transformed = data_transformed[-test.shape[0]:]
        return train_transformed, test_transformed

    def apply_spectral(self, train, test):
        spectral = SpectralEmbedding(n_neighbors=20, n_components=self.n_component)
        data_transformed = spectral.fit_transform(np.concatenate([train, test], axis=0))
        train_transformed = data_transformed[0:train.shape[0]]
        test_transformed = data_transformed[-test.shape[0]:]
        return train_transformed, test_transformed

    def apply_tsne(self, train, test):
        tsne = TSNE(n_components=self.n_component, perplexity=30, early_exaggeration=12, learning_rate=200.0, n_iter=1000)
        data_transformed = tsne.fit_transform(np.concatenate([train, test], axis=0))
        train_transformed = data_transformed[0:train.shape[0]]
        test_transformed = data_transformed[-test.shape[0]:]
        return train_transformed, test_transformed

    def apply_umap(self, train, test):
        umap = UMAP(n_components=self.n_component, init='random', random_state=0)
        data_transformed = umap.fit_transform(np.concatenate([train, test], axis=0))
        train_transformed = data_transformed[0:train.shape[0]]
        test_transformed = data_transformed[-test.shape[0]:]
        return train_transformed, test_transformed

    def apply_tsvd(self, train, test):
        tsvd = TruncatedSVD(n_components=self.n_component, init='random', random_state=0, algorithm='randomized'),
        data_transformed = tsvd.fit_transform(np.concatenate([train, test], axis=0))
        train_transformed = data_transformed[0:train.shape[0]]
        test_transformed = data_transformed[-test.shape[0]:]
        return train_transformed, test_transformed

    def apply_gm(self, train, test):
        gm = GaussianMixture(n_components=self.n_component, covariance_type='full', random_state=0)
        data_transformed = gm.fit(np.concatenate([train, test], axis=0))