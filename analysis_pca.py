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

from pca_analysis import PCAnalysis
from preprocessing import ylabel_encoding
from preprocessing.ylabel_encoding import YLabelEncoder
from subject_pca_analysis import SubjectPCsAnalysis
from visualization.matplotly_plot import plot_pcs_segment_profile, plot_pcs_mode_profile, plot_pcs_comp_profile, \
    plot_range_of_each_pc_mode, plot_pcs_variance_ratio, plot_bar_pc1s_pc2s, \
    plot_clustermap_pc1s_pc2s, plot_heatmap
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
    kihadataset_train = kihadataset_handler.run_combine_train_test_dataset()

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

        # kihadataset_train_all['y'] = kihadataset_train_all['y'][kihadataset_train_all['labels'][~(kihadataset_train_all['labels']['subjects'].isin(['S10', 'S21']))].index.to_list()]
        # kihadataset_train_all['labels'] = kihadataset_train_all['labels'][~(kihadataset_train_all['labels']['subjects'].isin(['S10', 'S21']))].reset_index(drop=True)
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

    config['risk_factor_activity'][1:3] = ['Stair Ascent', 'Stair Descent']

    # reformat the kinematic data from 3d to 2d
    data_labels = kihadataset_train_all['labels']
    data_y = np.reshape(kihadataset_train_all['y'],
                         [kihadataset_train_all['y'].shape[0],
                          kihadataset_train_all['y'].shape[1] * kihadataset_train_all['y'].shape[2]], 'F')
    dof_labels = config['selected_opensim_labels']
    data = data_y
    grouped_subject_speed_pcs = data_labels
    plt.figure()
    n_kinematic = len(dof_labels)
    c = 2
    # plt.figure()
    # s10_f = grouped_subject_speed_pcs[
    #     (grouped_subject_speed_pcs['subjects'] == 'S10') & (grouped_subject_speed_pcs['speed'] == 'fast')].index.values
    # s10_s = grouped_subject_speed_pcs[
    #     (grouped_subject_speed_pcs['subjects'] == 'S10') & (grouped_subject_speed_pcs['speed'] == 'slow')].index.values
    # s10_n = grouped_subject_speed_pcs[(grouped_subject_speed_pcs['subjects'] == 'S10') & (
    #             grouped_subject_speed_pcs['speed'] == 'normal')].index.values
    # s21_f = grouped_subject_speed_pcs[
    #     (grouped_subject_speed_pcs['subjects'] == 'S21') & (grouped_subject_speed_pcs['speed'] == 'fast')].index.values
    # s21_s = grouped_subject_speed_pcs[
    #     (grouped_subject_speed_pcs['subjects'] == 'S21') & (grouped_subject_speed_pcs['speed'] == 'slow')].index.values
    # s21_n = grouped_subject_speed_pcs[(grouped_subject_speed_pcs['subjects'] == 'S21') & (
    #             grouped_subject_speed_pcs['speed'] == 'normal')].index.values
    # s21 = grouped_subject_speed_pcs[grouped_subject_speed_pcs['subjects'] == 'S21'].index.values
    # s33 = grouped_subject_speed_pcs[grouped_subject_speed_pcs['subjects'] == 'S33'].index.values
    # s35 = grouped_subject_speed_pcs[grouped_subject_speed_pcs['subjects'] == 'S35'].index.values
    # for i, kinematic_label in enumerate(dof_labels):
    #     plt.subplot(round(n_kinematic / c), c, i + 1)
    #     for j in range(len(data)):
    #         len_data = range(i * range_len, i * range_len + range_len)
    #         if j in s10_f:
    #             plt.plot(data[j][len_data], c='r')
    #         elif j in s10_s:
    #             plt.plot(data[j][len_data], c='g')
    #         elif j in s10_n:
    #             plt.plot(data[j][len_data], c='b')
    #         # elif j in s35:
    #         #     plt.plot(data[j][len_data], c='c')
    #         # else:
    #         #     plt.plot(data[j][len_data], c='gray')
    # plt.show()

    kinematic_columns = [str(i) for i in range(data_y.shape[1])]
    data_df = pd.concat([data_labels, pd.DataFrame(data_y, columns=kinematic_columns)], axis=1)


    # generate encoder label for knee
    le_k = LabelEncoder()
    le_knee = le_k.fit(list(kihadataset_train_all['labels']['KNEE Status']))
    y_label_knee = le_knee.transform(list(kihadataset_train_all['labels']['KNEE Status']))
    y_label_knee[np.where(y_label_knee==2)] = 0
    # generate encoder label for Gender
    le_g = LabelEncoder()
    le_gender = le_g.fit(list(kihadataset_train_all['labels']['Gender (M/F)']))
    y_label_gender = le_gender.transform(list(kihadataset_train_all['labels']['Gender (M/F)']))
    le_gender.inverse_transform([0, 1])
    # generate encoder label for Speed
    le_s = LabelEncoder()
    le_speed = le_s.fit(list(kihadataset_train_all['labels']['speed']))
    y_label_speed = le_speed.transform(list(kihadataset_train_all['labels']['speed']))
    y_label = np.array([y_label_knee, y_label_speed, y_label_gender]).T
    data_df = pd.concat([data_df, pd.DataFrame(y_label, columns=['Knee Status', 'Speed Status', 'Gender Status'])], axis=1)

    aggregator_headers = {}
    patient_demographic_headers = ['Height (CM)', 'weight (KG)', 'SS %', 'P %', 'FDL %', 'FSR %', 'QL %', 'KOOS']
    patient_demographic_headers = ['Height (CM)', 'weight (KG)', 'KOOS']
    subject_headers = ['Knee Status', 'Speed Status', 'Gender Status', 'Height (CM)', 'weight (KG)', 'SS %', 'P %', 'FDL %', 'FSR %', 'QL %', 'KOOS']
    subject_headers = ['Knee Status', 'Speed Status', 'Gender Status', 'Height (CM)', 'weight (KG)', 'KOOS']
    selected_headers = subject_headers.copy()
    selected_headers.extend(kinematic_columns)
    for key in selected_headers:
        aggregator_headers[key] = 'mean'

    # config['risk_factor_activity'] = config['risk_factor_activity'][1:]
    for activity in config['risk_factor_activity']:
        data_df_activity = data_df[data_df['activity'] == activity]
        grouped_subject_speed = data_df_activity.groupby(['subjects', 'speed']).agg(aggregator_headers)
        kinematics = grouped_subject_speed[kinematic_columns].values
        patient_variables = grouped_subject_speed[patient_demographic_headers].values
        '--------------------------------------------'
        '------------------PCA Class-----------------'
        pcanalysis_handler = PCAnalysis(config, kinematics, 100, pc_variance=0.95)
        pcanalysis_handler.apply_pca()
        pcanalysis_handler.display_pca_variance_ratio()
        pcanalysis_handler.form_mode_variations(n_mode=4, std_factor=2)
        pcanalysis_handler.display_pcs_mode_profile(n_mode=4, std_factor=2, title='title')
        pcanalysis_handler.display_pcs_comp_profile(title='sss', abs_pcs_status=True)
        pcanalysis_handler.display_4linkage_mechanism(n_mode=4, std_factor=2, title='title')
        '--------------------------------------------'
        # subjectPCsAnalysis_handler = SubjectPCsAnalysis(config, patient_variables, kinematics, 100, pc_n_component = None, pc_variance = 0.95,
        # pc_subject_n_component = None, pc_subject_variance = 0.95)

        pca = PCA(0.95).fit(kinematics)
        cov = pca.get_covariance()
        pca_comp = pca.components_  # [n_component, n_features]  if don't chose any pca component or variance, then we will get n_component=n_sampel
        pca_comp_abs = abs(pca_comp)
        pca_mean = pca.mean_  # [n_features,1] mean over features
        pca_variance = pca.explained_variance_  # [n_component,1] variance of each component
        pca_variance_ratio = pca.explained_variance_ratio_  # [n_component,1] Percentage of variance explained by each of the selected components.
        # pca varinace bar plot
        plot_pcs_variance_ratio(pca_variance, pca_variance_ratio)

        # 3) calculate pca transform
        kinematics_pca_transformed = pca.transform(kinematics)


        # form a new data including the patient variables and new pcs scores
        patient_pcs = np.concatenate([patient_variables, kinematics_pca_transformed], axis=1)
        scaler = StandardScaler()
        scaler_fit = scaler.fit(patient_pcs)
        patient_pcs_scaled = scaler_fit.transform(patient_pcs)
        # standrize the patient_pcs data
        scaler = StandardScaler()
        scaler_fit = scaler.fit(patient_variables)
        patient_variables_scaled = scaler_fit.transform(patient_variables)
        scaler_pcs = StandardScaler()
        scaler_fit_pcs = scaler_pcs.fit(kinematics_pca_transformed.reshape(-1,1))
        kinematics_pca_transformed_scaled = scaler_fit_pcs.transform(kinematics_pca_transformed.reshape(-1,1))
        kinematics_pca_transformed_scaled = np.reshape(kinematics_pca_transformed_scaled,
                                        [kinematics_pca_transformed.shape[0], kinematics_pca_transformed.shape[1]],  'F')
        patient_pcs_scaled = np.concatenate([patient_variables_scaled, kinematics_pca_transformed_scaled], axis=1)

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
        plot_heatmap(pca_comp2_df, xlabel='Input Variables of PC2 (patients + PCs calculated from previous step (Kinematics Variable--> PCs)',
                     ylabel='PC2s Component Values')
        plot_bar_pc1s_pc2s(pca_comp2_df)
        # plot_clustermap_pc1s_pc2s(pca_comp2_df)



        # # multi linear regression between PCs and patient variable???
        # model = linear_model.LinearRegression()
        # x_scaled = patient_pcs_scaled[:, patient_variables.shape[1]:]
        # y_scaled = patient_pcs_scaled[:, np.arange(patient_variables.shape[1])]
        # y = patient_variables
        # clf = MultiOutputRegressor(model).fit(x_scaled, y_scaled)
        # reg_score = clf.score(x_scaled, y_scaled)
        # predict = clf.predict(x_scaled)
        # coef = np.array([clf.estimators_[i].coef_ for i in range(len(clf.estimators_))]).T
        # # form a data frame
        # coef_df = pd.DataFrame(coef, columns=patient_demographic_headers, index=pcs1_columns)
        # # plot heatmap
        # plt.figure()
        # sns.heatmap(coef_df, cmap="coolwarm", annot=True, fmt='.1g')
        # plt.ylabel('Output Variables of Linear Regression - PCs calculated from previous step (Kinematics Variable--> PCs)')
        # plt.xlabel('Input Variable of Linear Regression - Patient Variable')
        # plt.tight_layout()
        #
        # # define the direct multioutput wrapper model
        # wrapper = MultiOutputRegressor(model)
        # # define the evaluation procedure
        # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        # # evaluate the model and collect the scores
        # n_scores = cross_val_score(wrapper, x_scaled, y_scaled, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # accuracy_scores = cross_validate(wrapper, x_scaled, y_scaled, scoring=('r2', 'neg_mean_squared_error'), cv=cv,
        #                            return_train_score=True)
        # # force the scores to be positive
        # n_scores = absolute(n_scores)
        # # summarize performance
        # print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
        # print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))



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

        knee = []
        for i in grouped_subject_speed_pcs['Knee Status']:
            knee.append(le_knee.inverse_transform([int(i)])[0])
        gender = []
        for i in grouped_subject_speed_pcs['Gender Status']:
            gender.append(le_gender.inverse_transform([int(i)])[0])
        grouped_subject_speed_pcs['knee'] = knee
        grouped_subject_speed_pcs['knee'] = grouped_subject_speed_pcs['knee'].astype('category')
        grouped_subject_speed_pcs['gender'] = gender
        grouped_subject_speed_pcs['gender'] = grouped_subject_speed_pcs['gender'].astype('category')

        # selected_headers = ['Knee Status', 'Gender Status', 'Height (CM)', 'weight (KG)', 'KOOS', 'subject', 'speed', 'knee', 'gender']
        # selected_headers.extend((pcs_columns))
        # grouped_subject_speed_pcs[selected_headers].to_csv('gait_pcs.csv')

        corr_columns = subject_headers.extend(pcs_columns)
        corr_X_df = grouped_subject_speed_pcs[subject_headers].corr(method='pearson')
        # plot heatmap pearson correlation
        plt.figure(figsize=(10, 8))
        X_df_lt = corr_X_df.where(np.tril(np.ones(corr_X_df.shape)).astype(np.bool))
        sns.heatmap(X_df_lt, cmap="coolwarm", annot=True, fmt='.1g')
        plt.tight_layout()
        #
        # plt.figure(figsize=(10, 8))
        # sns.clustermap(corr_X_df, cmap="coolwarm", annot=True, fmt='.1g')
        # plt.tight_layout()
        # grouped_subject_speed_pcs.plot.scatter(x='Height (CM)', y='pc1', c='speed', colormap='viridis')
        # plt.figure()
        # grouped_subject_speed_pcs.plot.scatter(x='Height (CM)', y='pc2', c='speed', colormap='viridis')
        # plt.figure()
        # # pair plot speed
        # selected_headers = ['speed', 'Height (CM)', 'weight (KG)', 'KOOS']
        # selected_headers.extend((pcs_columns))
        # sns.pairplot(grouped_subject_speed_pcs[selected_headers], hue="speed", corner=True)
        # plt.tight_layout()
        # # pair plot knee
        # selected_headers = ['Knee Status', 'Height (CM)', 'weight (KG)', 'KOOS']
        # selected_headers.extend((pcs_columns))
        # sns.pairplot(grouped_subject_speed_pcs[selected_headers], hue="Knee Status", corner=True, palette="Set2")
        # plt.tight_layout()
        # # # pair plot subject
        # selected_headers = ['subject', 'Height (CM)', 'weight (KG)', 'KOOS']
        # selected_headers.extend((pcs_columns))
        # sns.pairplot(grouped_subject_speed_pcs[selected_headers], hue="subject", corner=True, palette="Set2", diag_kind="hist")
        # plt.tight_layout()
        # # pair plot gender
        # selected_headers = ['Gender Status', 'Height (CM)', 'weight (KG)', 'KOOS']
        # selected_headers.extend((pcs_columns))
        # sns.pairplot(grouped_subject_speed_pcs[selected_headers], hue="Gender Status", corner=True, palette="Set1")
        # plt.tight_layout()
        # # filter based subject
        # # Search_for_These_values = ['S09', 'S10', 'S11']
        # # pattern = '|'.join(Search_for_These_values)
        # # grouped_subject_speed_pcs_filter = grouped_subject_speed_pcs[selected_headers].loc[(grouped_subject_speed_pcs[selected_headers]['subject'].str.contains(pattern, case=False))]
        # # sns.pairplot(grouped_subject_speed_pcs_filter, hue="subject", corner=True, palette="Set2", diag_kind="hist")
        # # plt.tight_layout()
        # high_pc1_subjects = grouped_subject_speed_pcs[selected_headers]['subject'][grouped_subject_speed_pcs[selected_headers]['pc1']>=250]
        # high_pc1_subjects = grouped_subject_speed_pcs[selected_headers]['subject'][
        #     grouped_subject_speed_pcs[selected_headers]['pc2'] >= 250]
        # high_pc3_subjects = grouped_subject_speed_pcs[selected_headers]['subject'][
        #     grouped_subject_speed_pcs[selected_headers]['pc3'] >= 170]


        from scipy.stats import ttest_rel
        from scipy.stats import ttest_ind
        selected_headers = ['Knee Status', 'Gender Status', 'Height (CM)', 'weight (KG)', 'KOOS']
        selected_headers.extend((pcs_columns))
        aggregator_headers_test = ['mean', 'std', 'max', 'min']
        grouped_subject_speed_pcs[selected_headers].groupby(['Knee Status']).agg(aggregator_headers_test)
        b = grouped_subject_speed_pcs[selected_headers]['pc1'][grouped_subject_speed_pcs[selected_headers]['Knee Status'] == 0]
        a = grouped_subject_speed_pcs[selected_headers]['pc1'][grouped_subject_speed_pcs[selected_headers]['Knee Status'] == 1]
        equal_var = True
        result = ttest_ind(a, b, equal_var=equal_var)
        print('------------------------------')
        print('Equal variance between two sample:{}'.format(equal_var))
        print('PC1 difference between OA and TKA')
        print('calculated t-statistic = {}, p-value = {}'.format(result[0], result[1]))
        equal_var = False
        result = ttest_ind(a, b, equal_var=equal_var)
        print('------------------------------')
        print('Equal variance between two sample:{}'.format(equal_var))
        print('PC1 difference between OA and TKA')
        print('calculated t-statistic = {}, p-value = {}'.format(result[0], result[1]))


        b = grouped_subject_speed_pcs[selected_headers]['KOOS'][grouped_subject_speed_pcs[selected_headers]['Knee Status'] == 0]
        a = grouped_subject_speed_pcs[selected_headers]['KOOS'][grouped_subject_speed_pcs[selected_headers]['Knee Status'] == 1]
        equal_var = True
        result = ttest_ind(a, b, equal_var=equal_var)
        print('------------------------------')
        print('Equal variance between two sample:{}'.format(equal_var))
        print('KOOS difference between OA and TKA')
        print('calculated t-statistic = {}, p-value = {}'.format(result[0], result[1]))
        equal_var = False
        result = ttest_ind(a, b, equal_var=equal_var)
        print('------------------------------')
        print('Equal variance between two sample:{}'.format(equal_var))
        print('KOOS difference between OA and TKA')
        print('calculated t-statistic = {}, p-value = {}'.format(result[0], result[1]))
        '''
        statisticfloat or array
        The calculated t-statistic.
        
        pvaluefloat or array
        The two-tailed p-value.
        '''
        # grouped_subject_speed_pcs[selected_headers].to_csv('gait.csv')


        # mean of PC scores
        mean_score = np.mean(kinematics_pca_transformed, 0)
        # Std of PC scores
        std_score = np.std(kinematics_pca_transformed, 0)
        n_mode_variation = 13
        std_factor = 2
        mode_variation = {}
        for i in range(n_mode_variation):
            mode_std = np.mean(kinematics, 0) + (mean_score[i] + std_factor * std_score[i]) * pca_comp[i, :]
            mode_nstd = np.mean(kinematics, 0) + (mean_score[i] - std_factor * std_score[i]) * pca_comp[i, :]
            mode_variation['mode + {}stdev_pc{}'.format(std_factor, i+1)] = mode_std
            mode_variation['mode - {}stdev_pc{}'.format(std_factor, i+1)] = mode_nstd

        # plot Subject with high PC1, PC2, and PC3
        dof_labels =  config['selected_opensim_labels']
        data = kinematics
        plt.figure()
        n_kinematic = len(dof_labels)
        c = 2
        s10 = grouped_subject_speed_pcs[grouped_subject_speed_pcs['subject'] =='S10'].index.values
        s21= grouped_subject_speed_pcs[grouped_subject_speed_pcs['subject'] == 'S21'].index.values
        s33 = grouped_subject_speed_pcs[grouped_subject_speed_pcs['subject'] == 'S33'].index.values
        s35 = grouped_subject_speed_pcs[grouped_subject_speed_pcs['subject'] == 'S35'].index.values
        for i, kinematic_label in enumerate(dof_labels):
            plt.subplot(round(n_kinematic / c), c, i + 1)
            for j in range(len(data)):
                len_data = range(i * range_len, i * range_len + range_len)
                if j in s10:
                    plt.plot(data[j][len_data], c='r')
                elif j in s21:
                    plt.plot(data[j][len_data], c='g')
                elif j in s33:
                    plt.plot(data[j][len_data], c='b')
                elif j in s35:
                    plt.plot(data[j][len_data], c='c')
                else:
                    plt.plot(data[j][len_data], c='gray')
        plt.show()

        # plot pc mode profiles
        plot_pcs_mode_profile(kinematics, config['selected_opensim_labels'], activity + ' '+selected_side + '|' + 'affected_knee:' + str(affected_knee),
                              mode_variation, range_len, n_mode=n_mode_variation)
        plot_pcs_mode_profile(kinematics, config['selected_opensim_labels'], activity + ' '+selected_side + '|' + 'affected_knee:' + str(affected_knee),
                              mode_variation, range_len, n_mode=4)
        # plot range
        plot_range_of_each_pc_mode(config['selected_opensim_labels'],
                                             activity + ' ' + selected_side + '|' + 'affected_knee:' + str(affected_knee),
                                             pd.DataFrame.from_dict(mode_variation), range_len)
        # plot pcs component and pc abs component
        plot_pcs_comp_profile(config['selected_opensim_labels'], activity + ' '+selected_side + '|' + 'affected_knee:' + str(affected_knee),
                              pca_comp, range_len, abs_pcs=False)
        plot_pcs_comp_profile(config['selected_opensim_labels'], activity + ' '+selected_side + '|' + 'affected_knee:' + str(affected_knee),
                              pca_comp, range_len, abs_pcs=True)

        # plot pcs segment
        plot_pcs_segment_profile(kinematics, config['selected_opensim_labels'], activity + ' '+selected_side+ '|' + 'affected_knee:' + str(affected_knee),
                                 pca_comp_abs, range_len, n_pc_segment=None)
        plot_pcs_segment_profile(kinematics, config['selected_opensim_labels'], activity + ' '+selected_side+ '|' + 'affected_knee:' + str(affected_knee),
                                 pca_comp_abs, range_len, n_pc_segment=n_mode_variation)
        # plot 4 linkage mechanism
        linkage_mechanism_handler = Plot4LinkageMechanism(kinematics, config['selected_opensim_labels'],
                                                          activity + ' '+selected_side+ '|' + 'affected_knee:' + str(affected_knee) + '_'+ str(std_factor) + 'std',
                                                          mode_variation, range_len)
        linkage_mechanism_handler.plot_mechanism_row_basis(10, 60, 50, 20)

        plt.figure()
        colors = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.Set3(i) for i in range(10)]
        for i in range(len(pca_comp)):
            plt.plot(pca_comp[i, :], label='pc{}_score'.format(i+1), c=colors[i])
        plt.legend()
        plt.show()

        # plt.figure()
        # len_data = range(8 * range_len, 8 * range_len + range_len)
        # plt.plot(np.mean(kinematics, 0)[len_data], label='mean', c='k')
        # for i in range(len(pca_comp)):
        #     for j in range(len(pca_comp)-8):
        #         gait_data_2stdev_PC1 = np.mean(kinematics, 0) + (mean_score[i] + 2 * std_score[i]) * pca_comp[j, :]
        #         plt.plot(gait_data_2stdev_PC1[len_data], label='mean_{} std{} pc{}'.format(i, i, j))
        # plt.legend()
        # plt.show()
        a = 1


if __name__ == '__main__':
    run_main()
