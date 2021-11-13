# load data
import itertools
import seaborn as sns
import pickle as pk
import pandas as pd
import numpy as np
from pyDOE import ff2n
from scipy.stats import pearsonr
from sklearn import linear_model, tree, svm, neighbors, gaussian_process
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Exponentiation, RationalQuadratic, Matern
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, RandomForestRegressor, \
    StackingRegressor, ExtraTreesRegressor, RandomTreesEmbedding, VotingRegressor
import matplotlib.pyplot as plt
import pyDOE

def run_regressor():

    pcs = pd.read_csv('./result/all_pca_patient_variables_singlegait.csv')
    input_activity = "Gait"
    output_activity = 'Stair Ascent'
    activities = ["Gait", 'Stair Ascent', 'Stair Descent', 'STS']
    ia, oa, mn, rmses, mapes, r2s, rs = [], [], [], [], [], [], []
    current = 0
    model_save = True
    for input, output in itertools.combinations(activities, 2):
        for i in range(2):
            if i==0:
                input_activity= input
                output_activity = output
            if i==1:
                input_activity= output
                output_activity = input
            x = pcs[[c for c in pcs.columns if input_activity in c]].values
            y = pcs[[c for c in pcs.columns if output_activity in c]].values
            # standrize the patient_pcs data
            scaler = StandardScaler()
            # scaler = PolynomialFeatures(degree=2)
            # list of model
            models = {'LinearRegression': linear_model.LinearRegression(),
                            'RidgeCV': linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13)),
                            'Lasso': linear_model.Lasso(alpha=0.1),
                            'ElasticNet': linear_model.ElasticNet(),
                            'BayesianRidge': linear_model.BayesianRidge(),
                            # 'LogisticRegression':linear_model.LogisticRegression(),
                            # 'TweedieRegressor': linear_model.TweedieRegressor(power=1, alpha=0.5, link='log'),
                            # 'SGDRegressor': linear_model.SGDRegressor(max_iter=1000, tol=1e-3),
                            'DecisionTreeRegressor': tree.DecisionTreeRegressor(),
                            'SVMLinear' : svm.SVR(kernel='linear'),
                            'SVMPoly' : svm.SVR(kernel='poly', degree=2),
                            'SVMRbf' : svm.SVR(kernel='rbf'),
                            'KNeighborsRegressor ': neighbors.KNeighborsRegressor(n_neighbors=5),
                            # 'GaussianProcessRegressor': gaussian_process.GaussianProcessRegressor(kernel=RationalQuadratic),
                             }
            base_estimator = models['SVMLinear']
            ensemble_models = {
            "RandomForestRegressor": RandomForestRegressor(),
            "RandomTreesEmbedding": RandomTreesEmbedding(),
            "ExtraTreesRegressor": ExtraTreesRegressor(),
            'BaggingRegressor': BaggingRegressor(base_estimator=base_estimator),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "AdaBoostRegressor": AdaBoostRegressor()}
            # "VotingRegressor": VotingRegressor(base_estimator=base_estimator),
            # "StackingRegressor": StackingRegressor(base_estimator=base_estimator)}
            '''
            The class SGDRegressor implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties to fit linear regression models. 
            SGDRegressor is well suited for regression problems with a large number of training samples (> 10.000), for other problems we recommend Ridge, Lasso, or ElasticNet.
            '''
            for key, value in models.items():
                model_name = key
                model = value
                # model_name = 'LinearRegression'
                # model = models[model_name]
                model = MultiOutputRegressor(model)
                # model = ensemble_models['BaggingRegressor']
                pipeline = make_pipeline(scaler, model)
                cv = KFold(n_splits=len(y))
                accuracy_scores = cross_validate(pipeline, x, y,
                                                 scoring=('neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_absolute_percentage_error'), cv=cv,
                                                 return_train_score=False, return_estimator=True)

                y_pred = cross_val_predict(pipeline, x, y, cv=cv)
                mse = mean_squared_error(y, y_pred)
                rmse = mean_squared_error(y, y_pred, squared=False)
                mape = mean_absolute_percentage_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                r = np.mean([pearsonr(y[:, i], y_pred[:, i])[0] for i in range(y_pred.shape[1])])
                # save model
                trained_model = pipeline.fit(x, y)
                if model_save:
                    pk.dump(trained_model, open('./cache/regression_models/model_'+ model_name +'_'+ input_activity
                                                +'_'+output_activity+'.pkl', 'wb'))
                # plt.figure()
                # for c in range(y.shape[1]):
                #     plt.scatter(y[:, c], y_pred[:, c], label='pc{}'.format(c))
                # plt.plot([min(y.min(), y_pred.min()), max(y.max(), y_pred.max())], [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())], 'k')
                # plt.ylabel('true')
                # plt.xlabel('pred')
                # plt.text(-1, -1, 'rmse:{}\n mape: {}\n r: {}'.format(rmse, mape, r), fontsize=20)
                # plt.title('Input Activity: {}\n output activity:{} \n Model: {}'.format(input_activity, output_activity, model_name))
                # plt.legend()
                # plt.show()
                ia.extend([input_activity])
                oa.extend([output_activity])
                mn.extend([model_name])
                rmses.extend([rmse])
                mapes.extend([mape])
                r2s.extend([r2])
                rs.extend([r])
    results = {'input_activity':ia,
            'output_activity': oa,
            'model_name': mn,
            'RMSE':rmses,
            'MAPE': mapes,
            'r2':r2s,
            'r':rs}
    results_df = pd.DataFrame.from_dict(results)

    for metric in ['RMSE', 'MAPE', 'r2', 'r']:
        fig, axes = plt.subplots(2, 2, figsize=[10,8])
        for i, activity in enumerate(activities):
            sns.barplot(ax=axes[i // 2, i % 2], x='output_activity', y=metric, hue='model_name',
                        data=results_df[results_df['input_activity'] == activity])
            axes[i // 2, i % 2].set_title('Input Activity: ' + activity, fontdict={'fontsize': 20})
            axes[i // 2, i % 2].set_ylabel(ylabel=metric, fontdict={'fontsize': 20})
            axes[i // 2, i % 2].set_xlabel(xlabel='Output Activities', fontdict={'fontsize': 20})
            if i != 1:
                axes[i // 2, i % 2].legend().set_visible(False)
            else:
                axes[i // 2, i % 2].legend(bbox_to_anchor=[1.2, 0.5],
                                           loc='center')
        plt.tight_layout()
    a= 1


if __name__ == '__main__':
    run_regressor()
    a =1
