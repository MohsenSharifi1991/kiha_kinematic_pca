import numpy as np
from sklearn.decomposition import PCA

from visualization.matplotly_plot import boxplot_pca_train_test, scatter_pca_train_test
from visualization.wandb_plot import wandb_scatter_2d_train_test


class PCAPlot:
    def __init__(self, kihadataset_train, kihadataset_test, n_pca, remove_outlier_state=True, boxplot=False, scatter=True):
        self.kihadataset_train = kihadataset_train
        self.kihadataset_test = kihadataset_test
        del kihadataset_train, kihadataset_test
        self.train_x = self.kihadataset_train['x']
        self.train_y = self.kihadataset_train['y']
        self.train_label = self.kihadataset_train['labels']
        self.test_x = self.kihadataset_test['x']
        self.test_y = self.kihadataset_test['y']
        self.test_label = self.kihadataset_test['labels']
        self.n_pca = n_pca
        self.boxplot = boxplot
        self.scatter = scatter
        self.remove_outlier_state = remove_outlier_state
        self.x_axis1 = self.train_x.shape[1]
        self.x_axis2 = self.train_x.shape[2]
        self.y_axis1 = self.train_y.shape[1]
        self.y_axis2 = self.train_y.shape[2]

    def run_pcaplot(self):
        self.run_outlier_plot()
        self.return_updated_data()
        return self.kihadataset_train, self.kihadataset_test

    def return_updated_data(self):
        self.reshape_2dto3d()
        self.kihadataset_train['x'] = self.train_x
        self.kihadataset_train['y'] = self.train_y
        self.kihadataset_train['labels'] = self.train_label
        self.kihadataset_test['x'] = self.test_x
        self.kihadataset_test['y'] = self.test_y
        self.kihadataset_test['labels'] = self.test_label

    def run_outlier_plot(self):
        self.reshape_3dto2d()
        self.train_x_pca, self.test_x_pca = self.apply_pca(self.train_x, self.test_x)
        self.train_y_pca, self.test_y_pca = self.apply_pca(self.train_y, self.test_y)
        if self.remove_outlier_state:
            self.train_x_pca, self.train_y_pca, self.train_x, self.train_y, self.train_label = self.remover_outlier(self.train_x_pca, self.train_y_pca,
                                                                                        self.train_x, self.train_y, self.train_label)
            self.test_x_pca, self.test_y_pca, self.test_x, self.test_y, self.test_label = self.remover_outlier(self.test_x_pca, self.test_y_pca,
                                                                                        self.test_x, self.test_y, self.test_label)
        if self.boxplot:
            boxplot_pca_train_test(self.train_x_pca, self.test_x_pca, self.train_y_pca, self.test_y_pca)
        if self.scatter:
            # scatter_pca_train_test(self.train_x_pca, self.test_x_pca, self.train_label, self.test_label, status='X')
            # scatter_pca_train_test(self.train_y_pca, self.test_y_pca, self.train_label, self.test_label, status='Y')
            wandb_scatter_2d_train_test(self.train_x_pca, self.test_x_pca, self.train_label, self.test_label, status_title='X')
            wandb_scatter_2d_train_test(self.train_y_pca, self.test_y_pca, self.train_label, self.test_label, status_title='Y')

    def reshape_3dto2d(self):
        self.train_x = np.reshape(self.train_x,
                             [self.train_x.shape[0], self.x_axis1*self.x_axis2])
        self.train_y = np.reshape(self.train_y,
                             [self.train_y.shape[0], self.y_axis1*self.y_axis2])
        self.test_x = np.reshape(self.test_x,
                            [self.test_x.shape[0], self.x_axis1*self.x_axis2])
        self.test_y = np.reshape(self.test_y,
                            [self.test_y.shape[0], self.y_axis1*self.y_axis2])

    def reshape_2dto3d(self):
        self.train_x = np.reshape(self.train_x,
                             [self.train_x.shape[0], self.x_axis1, self.x_axis2])
        self.train_y = np.reshape(self.train_y,
                             [self.train_y.shape[0], self.y_axis1, self.y_axis2])
        self.test_x = np.reshape(self.test_x,
                            [self.test_x.shape[0], self.x_axis1, self.x_axis2])
        self.test_y = np.reshape(self.test_y,
                            [self.test_y.shape[0], self.y_axis1, self.y_axis2])

    def apply_pca(self, train, test):
        pca = PCA(n_components=self.n_pca).fit(train)
        train_pca = pca.transform(train)
        test_pca = pca.transform(test)
        return train_pca, test_pca

    def remover_outlier(self, x_pca, y_pca, x, y, label):
        # outlier_index = np.where((x_pca[:, 0:5] < -200) | (x_pca[:, 0:5] > 35))[0]
        outlier_x =[]
        for i in range(x_pca.shape[1]):
            outlier_x.append(self.outlier_detection(x_pca[:, i])[0])
        outlier_index = np.unique(np.concatenate(outlier_x))

        x_pca = np.delete(x_pca, outlier_index, axis=0)
        y_pca = np.delete(y_pca, outlier_index, axis=0)
        x = np.delete(x, outlier_index, axis=0)
        y = np.delete(y, outlier_index, axis=0)
        label = label.drop(index=outlier_index).reset_index(drop=True)
        return x_pca, y_pca, x, y, label

    def outlier_detection(self, data):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (iqr * 1.5)
        upper_bound = q3 + (iqr * 1.5)
        return np.where((data < lower_bound) | (data > upper_bound))