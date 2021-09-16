import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from numpy import absolute
from sklearn.model_selection import RepeatedKFold, cross_val_score, cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from config import get_config
from kihadataset import KIHADataSet
import matplotlib.pyplot as plt
import pandas as pd

from visualization.matplotly_plot import plot_pcs_segment_profile, plot_pcs_mode_profile
from visualization.plot_4linkage_mechanism import Plot4LinkageMechanism
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
import seaborn as sns


def run_main():
    np.random.seed(42)
    config = get_config()
    range_len = config['target_padding_length']

    # build and split kiha dataset to training and test
    kihadataset_handler = KIHADataSet(config)
    kihadataset_train, kihadataset_test = kihadataset_handler.run_dataset_split()

    # pick data for only one side: R or L
    kihadataset_train_all = kihadataset_train.copy()
    kihadataset_train_all['labels']['tka_side'] = len(kihadataset_train_all['labels']) * ['Nan']
    kihadataset_train_all['labels']['tka_side'][kihadataset_train_all['labels']['KNEE Status']=='BiTKA'] = 'BiTKA'
    kihadataset_train_all['labels']['tka_side'][(kihadataset_train_all['labels']['KNEE Status'] == 'TKA') &
                                                (kihadataset_train_all['labels']['1-Knee Implant L'] == 'Yes')] = 'LTKA'
    kihadataset_train_all['labels']['tka_side'][(kihadataset_train_all['labels']['KNEE Status'] == 'TKA') &
                                                (kihadataset_train_all['labels']['1-Knee Implant R'] == 'Yes')] = 'RTKA'
    # left or side selection method:
    # 1) Left or Right
    selected_side = 'R'
    affected_knee = False
    if affected_knee!=True:
        kihadataset_train_all['y'] = kihadataset_train_all['y'][kihadataset_train_all['labels'][kihadataset_train_all['labels']['side_seg']==selected_side].index.to_list()]
        kihadataset_train_all['labels'] = kihadataset_train_all['labels'][kihadataset_train_all['labels']['side_seg'] == selected_side].reset_index(drop=True)
    else:
        pass
        # 2) Left or Right default and Affected TKA side
        # kihadataset_train_all['y'] = kihadataset_train_all['y'][kihadataset_train_all['labels'][(
        #     ((kihadataset_train_all['labels']['tka_side'] =='LTKA') & (kihadataset_train_all['labels']['side_seg']=='L')) |
        #     ((kihadataset_train_all['labels']['tka_side'] == 'RTKA') & (kihadataset_train_all['labels']['side_seg'] == 'R')) |
        #     ((kihadataset_train_all['labels']['tka_side'] == 'BiTKA') & (kihadataset_train_all['labels']['side_seg'] == selected_side)) |
        #      ((kihadataset_train_all['labels']['tka_side'] == 'Nan') & (kihadataset_train_all['labels']['side_seg'] == selected_side))
        # )].index.to_list()]
        # kihadataset_train_all['labels'] = kihadataset_train_all['labels'][(
        #         ((kihadataset_train_all['labels']['tka_side'] == 'LTKA') & (
        #                     kihadataset_train_all['labels']['side_seg'] == 'L')) |
        #         ((kihadataset_train_all['labels']['tka_side'] == 'RTKA') & (
        #                     kihadataset_train_all['labels']['side_seg'] == 'R')) |
        #         ((kihadataset_train_all['labels']['tka_side'] == 'BiTKA') & (
        #                     kihadataset_train_all['labels']['side_seg'] == selected_side)) |
        #         ((kihadataset_train_all['labels']['tka_side'] == 'Nan') & (
        #                     kihadataset_train_all['labels']['side_seg'] == selected_side))
        # )].reset_index(drop=True)

    # rename stair up-down to stair up and stair down
    kihadataset_train_all['labels']['activity'][(kihadataset_train_all['labels']['activity'].str.contains('Stair')) &
                                        (kihadataset_train_all['labels']['trial_num'].str.contains('_1'))] = 'Stair Up'
    kihadataset_train_all['labels']['activity'][(kihadataset_train_all['labels']['activity'].str.contains('Stair')) &
                                        (kihadataset_train_all['labels']['trial_num'].str.contains('_2'))] = 'Stair Down'
    config['risk_factor_activity'][1:3] = ['Stair Up', 'Stair Down']

    # reformat the kinematic data from 3d to 2d
    data_labels = kihadataset_train_all['labels']
    data_y = np.reshape(kihadataset_train_all['y'],
                         [kihadataset_train_all['y'].shape[0],
                          kihadataset_train_all['y'].shape[1] * kihadataset_train_all['y'].shape[2]], 'F')
    kinematic_columns = [str(i) for i in range(data_y.shape[1])]
    data_df = pd.concat([data_labels, pd.DataFrame(data_y, columns=kinematic_columns)], axis=1)

    # generate encoder label for knee
    le = LabelEncoder()
    le_knee = le.fit(list(kihadataset_train_all['labels']['KNEE Status']))
    y_label_knee = le_knee.transform(list(kihadataset_train_all['labels']['KNEE Status']))
    y_label_knee[np.where(y_label_knee==2)] = 0
    # generate encoder label for Speed
    le_speed = le.fit(list(kihadataset_train_all['labels']['speed']))
    y_label_speed = le_speed.transform(list(kihadataset_train_all['labels']['speed']))
    y_label = np.array([y_label_knee, y_label_speed]).T
    data_df = pd.concat([data_df, pd.DataFrame(y_label, columns=['Knee Status', 'Speed Status'])], axis=1)

    aggregator_headers = {}
    patient_demographic_headers = ['Height (CM)', 'weight (KG)', 'SS %', 'P %', 'FDL %', 'FSR %', 'QL %', 'KOOS']
    patient_demographic_headers = ['Height (CM)', 'weight (KG)', 'KOOS']
    subject_headers = ['Knee Status', 'Speed Status', 'Height (CM)', 'weight (KG)', 'SS %', 'P %', 'FDL %', 'FSR %', 'QL %', 'KOOS']
    subject_headers = ['Knee Status', 'Speed Status', 'Height (CM)', 'weight (KG)', 'KOOS']
    selected_headers = subject_headers.copy()
    selected_headers.extend(kinematic_columns)
    for key in selected_headers:
        aggregator_headers[key] = 'mean'

    for activity in config['risk_factor_activity']:
        data_df_activity = data_df[data_df['activity'] == activity]
        grouped_subject_speed = data_df_activity.groupby(['subjects', 'speed']).agg(aggregator_headers)
        kinematics = grouped_subject_speed[kinematic_columns].values
        patient_variables = grouped_subject_speed[patient_demographic_headers].values

        pca = PCA(0.95).fit(kinematics)
        cov = pca.get_covariance()
        pca_comp = pca.components_  # [n_component, n_features]  if don't chose any pca component or variance, then we will get n_component=n_sampel
        pca_comp_abs = abs(pca_comp)
        pca_mean = pca.mean_  # [n_features,1] mean over features
        pca_variance = pca.explained_variance_  # [n_component,1] variance of each component
        pca_variance_ratio = pca.explained_variance_ratio_  # [n_component,1] Percentage of variance explained by each of the selected components.
        # 3) calculate pca transform
        kinematics_pca_transformed = pca.transform(kinematics)


        # form a new data including the patient variables and new pcs scores
        patient_pcs = np.concatenate([patient_variables, kinematics_pca_transformed], axis=1)
        # standrize the patient_pcs data
        scaler = StandardScaler()
        scaler_fit = scaler.fit(patient_pcs)
        patient_pcs_scaled = scaler_fit.transform(patient_pcs)
        # run PCs on the scaled data
        pca2 = PCA(0.95).fit(patient_pcs_scaled)
        cov2 = pca2.get_covariance()
        pca_comp2 = pca2.components_  # [n_component, n_features]  if don't chose any pca component or variance, then we will get n_component=n_sampel
        pca_comp_abs2 = abs(pca_comp2)
        # 3) calculate pca transform
        patient_pcs_scaled_transformed = pca2.transform(patient_pcs_scaled)
        pcs2_columns = ['pc2_' + str(i + 1) for i in range(patient_pcs_scaled_transformed.shape[1])]
        pcs1_columns = ['pc1_' + str(i + 1) for i in range(kinematics_pca_transformed.shape[1])]
        pca_comp2_df = pd.DataFrame(pca_comp2, columns=patient_demographic_headers + pcs1_columns, index=pcs2_columns)
        sns.heatmap(pca_comp2_df, cmap="coolwarm", annot=True, fmt='.1g')
        plt.xlabel('Input Variables of PC2 (patients + PCs calculated from previous step (Kinematics Variable--> PCs)')
        plt.ylabel('PC2s Component Values')
        plt.tight_layout()


        pca_comp2_df['pc2s'] = pca_comp2_df.index
        pca_comp2_df2 = pca_comp2_df.reset_index(drop=True)
        pca_comp2_df2 = pca_comp2_df2.melt(id_vars=['pc2s'])
        plt.figure()
        sns.barplot(x='variable', y='value', hue='pc2s', data=pca_comp2_df2, palette="tab20")
        plt.xlabel('Input Variables of PC2 (patients + PCs calculated from previous step (Kinematics Variable--> PCs)')
        plt.ylabel('PCs Component Values')
        plt.tight_layout()

        # multi linear regression between PCs and patient variable???
        model = linear_model.LinearRegression()
        x_scaled = patient_pcs_scaled[:, patient_variables.shape[1]:]
        y_scaled = patient_pcs_scaled[:, np.arange(patient_variables.shape[1])]
        y = patient_variables
        clf = MultiOutputRegressor(model).fit(x_scaled, y_scaled)
        reg_score = clf.score(x_scaled, y_scaled)
        predict = clf.predict(x_scaled)
        coef = np.array([clf.estimators_[i].coef_ for i in range(len(clf.estimators_))]).T
        # form a data frame
        coef_df = pd.DataFrame(coef, columns=patient_demographic_headers, index=pcs1_columns)
        # plot heatmap
        plt.figure()
        sns.heatmap(coef_df, cmap="coolwarm", annot=True, fmt='.1g')
        plt.ylabel('Output Variables of Linear Regression - PCs calculated from previous step (Kinematics Variable--> PCs)')
        plt.xlabel('Input Variable of Linear Regression - Patient Variable')
        plt.tight_layout()

        # define the direct multioutput wrapper model
        wrapper = MultiOutputRegressor(model)
        # define the evaluation procedure
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(wrapper, x_scaled, y_scaled, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        accuracy_scores = cross_validate(wrapper, x_scaled, y_scaled, scoring=('r2', 'neg_mean_squared_error'), cv=cv,
                                   return_train_score=True)
        # force the scores to be positive
        n_scores = absolute(n_scores)
        # summarize performance
        print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))






        pcs_columns = ['pc'+str(i+1) for i in range(kinematics_pca_transformed.shape[1])]
        grouped_subject_speed_pcs = pd.concat([grouped_subject_speed,
                                               pd.DataFrame(kinematics_pca_transformed, columns=pcs_columns, index=grouped_subject_speed.index)], axis=1)

        s = [grouped_subject_speed_pcs.index.values[s][0] for s in range(len(grouped_subject_speed_pcs))]
        p = [grouped_subject_speed_pcs.index.values[p][1] for p in range(len(grouped_subject_speed_pcs))]
        grouped_subject_speed_pcs['subject'] = s
        grouped_subject_speed_pcs['speed'] = p
        grouped_subject_speed_pcs['subject'] = grouped_subject_speed_pcs['subject'].astype('category')
        grouped_subject_speed_pcs['speed'] = grouped_subject_speed_pcs['speed'].astype('category')
        grouped_subject_speed_pcs = grouped_subject_speed_pcs.reset_index(drop=True)

        corr_columns = subject_headers.extend(pcs_columns)

        corr_X_df = grouped_subject_speed_pcs[subject_headers].corr(method='pearson')
        # plot heatmap pearson correlation
        plt.figure(figsize=(10, 8))
        X_df_lt = corr_X_df.where(np.tril(np.ones(corr_X_df.shape)).astype(np.bool))
        sns.heatmap(X_df_lt, cmap="coolwarm", annot=True, fmt='.1g')
        plt.tight_layout()

        # grouped_subject_speed_pcs.plot.scatter(x='Height (CM)', y='pc1', c='speed', colormap='viridis')
        # plt.figure()
        # grouped_subject_speed_pcs.plot.scatter(x='Height (CM)', y='pc2', c='speed', colormap='viridis')
        # plt.figure()

        # selected_headers = ['speed', 'Height (CM)', 'weight (KG)', 'KOOS']
        # selected_headers.extend((pcs_columns))
        # sns.pairplot(grouped_subject_speed_pcs[selected_headers], hue="speed", corner=True)
        # plt.tight_layout()
        #
        # selected_headers = ['Knee Status', 'Height (CM)', 'weight (KG)', 'KOOS']
        # selected_headers.extend((pcs_columns))
        # sns.pairplot(grouped_subject_speed_pcs[selected_headers], hue="Knee Status", corner=True, palette="Set2")
        # plt.tight_layout()


        # mean of PC scores
        mean_score = np.mean(kinematics_pca_transformed, 0)
        # Std of PC scores
        std_score = np.std(kinematics_pca_transformed, 0)
        n_mode_variation = 3
        std_factor = 2
        mode_variation = {}
        for i in range(n_mode_variation):
            mode_std = np.mean(kinematics, 0) + (mean_score[i] + std_factor * std_score[i]) * pca_comp[i, :]
            mode_nstd = np.mean(kinematics, 0) + (mean_score[i] - std_factor * std_score[i]) * pca_comp[i, :]
            mode_variation['mode + {}stdev_pc{}'.format(std_factor, i+1)] = mode_std
            mode_variation['mode- {}stdev_pc{}'.format(std_factor, i+1)] = mode_nstd

        # plot pcs123 profiles
        plot_pcs_mode_profile(kinematics, config['selected_opensim_labels'], activity + ' '+selected_side + '|' + 'affected_knee:' + str(affected_knee),
                              mode_variation, range_len, n_mode=n_mode_variation)
        # plot pcs segment
        plot_pcs_segment_profile(kinematics, config['selected_opensim_labels'], activity + ' '+selected_side+ '|' + 'affected_knee:' + str(affected_knee),
                                 pca_comp_abs, range_len, n_pc_segment=None)
        plot_pcs_segment_profile(kinematics, config['selected_opensim_labels'], activity + ' '+selected_side+ '|' + 'affected_knee:' + str(affected_knee),
                                 pca_comp_abs, range_len, n_pc_segment=3)
        # plot 4 linkage mechanism
        linkage_mechanism_handler = Plot4LinkageMechanism(kinematics, config['selected_opensim_labels'], activity + ' '+selected_side+ '|' + 'affected_knee:' + str(affected_knee),
                           mode_variation['mode + 2stdev_pc1'],  mode_variation['mode - 2stdev_pc1'],
                           mode_variation['mode + 2stdev_pc2'],  mode_variation['mode - 2stdev_pc2'],
                           mode_variation['mode + 2stdev_pc3'],  mode_variation['mode - 2stdev_pc3'],
                           range_len)
        linkage_mechanism_handler.plot_mechanism(10, 60, 50, 20)


        plt.figure()
        len_data = range(4 * range_len, 4 * range_len + range_len)
        plt.plot(np.mean(kinematics, 0)[len_data], label='mean', c='k')
        for i in range(len(pca_comp)):
            for j in range(len(pca_comp)-8):
                gait_data_2stdev_PC1 = np.mean(kinematics, 0) + (mean_score[i] + 2 * std_score[i]) * pca_comp[j, :]
                plt.plot(gait_data_2stdev_PC1[len_data], label='mean_{} std{} pc{}'.format(i, i, j))
        plt.legend()
        plt.show()
        a = 1



if __name__ == '__main__':
    run_main()
