from sklearn.decomposition import PCA


class PCAReduction:
    def __init__(self, kihadataset_train, kihadataset_test, variance=0.95):
        self.kihadataset_train = kihadataset_train
        self.kihadataset_test = kihadataset_test
        self.variance = variance
        self.return_updated_data()

    def return_updated_data(self):
        train_pca, test_pca = self.apply_pca(self.kihadataset_train['x'], self.kihadataset_test['x'])
        self.kihadataset_train['x'] = train_pca
        self.kihadataset_test['x'] = test_pca

    def apply_pca(self, train, test):
        pca = PCA(n_components=self.variance).fit(train)
        train_pca = pca.transform(train)
        test_pca = pca.transform(test)
        return train_pca, test_pca


