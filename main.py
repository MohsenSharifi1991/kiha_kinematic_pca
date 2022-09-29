import itertools
import matplotlib

from generator import run_generative_model
from sklearn.metrics import mean_squared_error
matplotlib.use('TKAgg')
import networkx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch
from numpy import absolute
from sklearn.model_selection import RepeatedKFold, cross_val_score, cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from config import get_config
from kihadataset import KIHADataSet
import matplotlib.pyplot as plt
import pandas as pd

from pca_analysis import PCAnalysis
from preprocessing.ylabel_encoding import YLabelEncoder
from subject_pca_analysis import SubjectPCsAnalysis
from timesetires_similarity import TimeSeriesSimilarity
from visualization.matplotly_plot import plot_pcs_segment_profile, plot_pcs_mode_profile, plot_pcs_comp_profile, \
    plot_range_of_each_pc_mode, plot_pcs_variance_ratio, plot_bar_pc1s_pc2s, \
    plot_clustermap_pc1s_pc2s, plot_scatter_pair, plot_heatmap, plot_kinematics_mean_std, plot_cluster_heatmap, \
    plot_box_plot_pcs_hue, plot_graph_network, plot_warping_kiha, plot_heatmap_demographic
from visualization.plot_4linkage_mechanism import Plot4LinkageMechanism
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
import wandb
import pickle as pk
import seaborn as sns
wandb.init(project='kia_kinematic_pca')

export_csv = False
export_xlsx = False
scatter_pairplot = False
pca_save = False
display_synthetic_kinematics = True

def run_main():
    np.random.seed(42)
    config = get_config()

    # build and split kiha dataset to training and test
    kihadataset_handler = KIHADataSet(config)
    kihadataset_train = kihadataset_handler.run_combine_train_test_dataset()
    kihadataset_train = kihadataset_handler.filter_data_based_on_side(kihadataset_train, selected_side='R')
    config['selected_opensim_labels'] = ["pelvis_tilt", "pelvis_list", "pelvis_rotation", "hip_flexion_r", "hip_adduction_r","hip_rotation_r","knee_angle_r", "ankle_angle_r"]
    # fill nan age: S19 TKA = Mean age of TKA
    kihadataset_train['labels']['Age'][kihadataset_train['labels']['Age'] == '?'] = round(kihadataset_train['labels']['Age'][(kihadataset_train['labels']['KNEE Status']=='TKA') & (kihadataset_train['labels']['Age']!='?')].mean())
    kihadataset_train['labels']['activity'][kihadataset_train['labels']['activity'] == 'STS 30Sec'] = 'STS'
    # reformat the kinematic data from 3d to 2d
    # data_y = np.vstack(kihadataset_train['y']) # for saving as csv to use in gan project
    # data_y_df = pd.DataFrame(data=data_y, columns=config['selected_opensim_labels'])
    # data_y_df.to_csv('kiha_kinematic_right.csv', index = False, header = True)
    data_labels = kihadataset_train['labels']
    # data_labels.to_csv('kiha_data_labels_right.csv', index=False, header=True)
    data_y = np.reshape(kihadataset_train['y'],
                         [kihadataset_train['y'].shape[0],
                          kihadataset_train['y'].shape[1] * kihadataset_train['y'].shape[2]], 'F')

    # form data df: kinematic and y labels
    kinematic_columns = [str(i) for i in range(data_y.shape[1])]
    data_df = pd.concat([data_labels, pd.DataFrame(data_y, columns=kinematic_columns)], axis=1)

    # encode y labels
    data_labels = data_labels.rename(columns={"KNEE Status": "knee", "Gender (M/F)": "gender"})
    ylabelencoder_handler = YLabelEncoder(data_labels, ['knee', 'gender', 'speed'])
    encoder_functions, y_label_df = ylabelencoder_handler.run_encoding()
    data_df = pd.concat([data_df, y_label_df], axis=1)
    # encode PT Change
    # data_df['pt risk status'] = pd.DataFrame(data=np.zeros([data_df.shape[0], 1]))
    # data_df['pt risk status'][data_df['subjects'].isin(['S19', 'S34'])] =1

    # form the aggregator headers
    aggregator_headers_mean = {}
    patient_demographic_headers = config['patient_demographic_headers']
    patient_demographic_headers = patient_demographic_headers.copy()
    patient_demographic_headers.extend(kinematic_columns)
    for key in patient_demographic_headers:
        aggregator_headers_mean[key] = 'mean'
    aggregator_headers_std = {}
    for key in patient_demographic_headers:
        aggregator_headers_std[key] = 'std'
    # run through each activity
    speed_status = False
    status =False
    pcs_activities_variance_ratio = {}
    mode_variations_activities = {}
    original_vs_synthetic = {'activity':[],'n_sample':[],
                             'r_oa':[], 'mse_oa':[],
                             'r_tka':[], 'mse_tka':[]}
    for activity in config['risk_factor_activity'][0:3]:
        data_df_activity = data_df[data_df['activity'] == activity].reset_index()
        if activity =='STS':
            pt_stand_risk_factor = pd.DataFrame(data=np.zeros([data_df_activity.shape[0], 1]),
                                                columns=['pt_stand_risk_factor'])
            pt_stand2flexseated_risk_factor = pd.DataFrame(data=np.zeros([data_df_activity.shape[0], 1]),
                                                columns=['pt_stand2flexseated_risk_factor'])
            pt_stand_risk_factor[(data_df_activity['phase_1_value_pt']<-10).values] = 1
            pt_stand2flexseated_risk_factor[(data_df_activity['phase_1_value_pt']-data_df_activity['phase_2_value_pt'] > 20).values] = 1
            data_df_activity['pt_stand_risk_factor'] = pt_stand_risk_factor['pt_stand_risk_factor']
            data_df_activity['pt_stand2flexseated_risk_factor'] = pt_stand2flexseated_risk_factor['pt_stand2flexseated_risk_factor']
        if speed_status:
            grouped_subject_speed = data_df_activity.groupby(['subjects', 'speed']).agg(aggregator_headers_mean)
        else:
            grouped_subject_speed = data_df_activity.groupby(['subjects']).agg(aggregator_headers_mean)
        kinematics = grouped_subject_speed[kinematic_columns].values
        patient_variables = grouped_subject_speed[config["patient_variable_headers"]].values
        # get OA vs TKA index
        grouped_subject_speed_index = grouped_subject_speed.copy().reset_index(drop=True)
        tka_oa_indx = {'tka_index': grouped_subject_speed_index[grouped_subject_speed_index['knee status']==0].index.values,
            'oa_index': grouped_subject_speed_index[grouped_subject_speed_index['knee status']==1].index.values}
        male_female_indx = {'female_index': grouped_subject_speed_index[grouped_subject_speed_index['gender status']==0].index.values,
            'male_index': grouped_subject_speed_index[grouped_subject_speed_index['gender status']==1].index.values}
        # pt_risk_indx = {'low_risk_index': grouped_subject_speed_index[grouped_subject_speed_index['pt risk status']==0].index.values,
        #     'high_risk_index': grouped_subject_speed_index[grouped_subject_speed_index['pt risk status']==1].index.values}
        if activity == 'Gait' and speed_status== True:
            slow_normal_fast_indx = {'slow_index': grouped_subject_speed_index[
                grouped_subject_speed_index['speed status'] == 2].index.values,
                                'normal_index': grouped_subject_speed_index[
                                    grouped_subject_speed_index['speed status'] == 1].index.values,
                                'fast_index': grouped_subject_speed_index[
                                    grouped_subject_speed_index['speed status'] == 0].index.values
                                }
            plot_kinematics_mean_std(kinematics, slow_normal_fast_indx, config['selected_opensim_labels'],
                                     activity, config['target_padding_length'],wandb_plot=True)
        # visualize OA vs TKA kinematics
        plot_kinematics_mean_std(kinematics, tka_oa_indx, config['y_axis_lims'], config['selected_opensim_labels'],
                                 activity, config['target_padding_length'], wandb_plot=True)


        if display_synthetic_kinematics:
            for n in range(10, 60, 10):
                n_sample = n
                oa_sample = run_generative_model(activity, 'oa', n_sample)
                tka_sample = run_generative_model(activity, 'tka', n_sample)
                synthetic_kinematics = np.concatenate([tka_sample, oa_sample])
                tka_oa_synthetic_indx = {'tka_synthetic': np.arange(0, len(oa_sample)), 'oa_synthetic': np.arange(len(oa_sample), len(synthetic_kinematics))}
                plot_kinematics_mean_std(synthetic_kinematics, tka_oa_synthetic_indx, config['y_axis_lims'], config['selected_opensim_labels'],
                                         activity + '_synthetic '+str(n_sample), config['target_padding_length'], wandb_plot=True)
                kinematics_original_synthetic = np.concatenate([kinematics, synthetic_kinematics])
                indexes = {'tka_original': tka_oa_indx['tka_index'],'oa_original': tka_oa_indx['oa_index'],
                           'tka_synthetic': np.arange(len(kinematics)+0, len(kinematics)+len(oa_sample)), 'oa_synthetic': np.arange(len(kinematics)+len(oa_sample), len(kinematics)+len(synthetic_kinematics))}
                plot_kinematics_mean_std(kinematics_original_synthetic, indexes, config['y_axis_lims'], config['selected_opensim_labels'],
                                         activity + '_original and synthetic {}'.format(n_sample), config['target_padding_length'], wandb_plot=True)
                original_oa = np.mean(kinematics[tka_oa_indx['oa_index']], 0)
                synthetic_oa = np.mean(oa_sample, 0)
                original_tka = np.mean(kinematics[tka_oa_indx['tka_index']], 0)
                synthetic_tka = np.mean(tka_sample, 0)
                r_oa = np.corrcoef(original_oa, synthetic_oa)[0,1]
                mse_oa= mean_squared_error(original_oa, synthetic_oa)
                r_tka = np.corrcoef(original_tka, synthetic_tka)[0,1]
                mse_tka = mean_squared_error(original_tka, synthetic_tka)
                original_vs_synthetic['activity'].append(activity)
                original_vs_synthetic['n_sample'].append(n_sample)
                original_vs_synthetic['r_oa'].append(r_oa)
                original_vs_synthetic['mse_oa'].append(mse_oa)
                original_vs_synthetic['r_tka'].append(r_tka)
                original_vs_synthetic['mse_tka'].append(mse_tka)

        # # visualize M vs F kinematics
        # plot_kinematics_mean_std(kinematics, male_female_indx, config['selected_opensim_labels'],
        #                          activity, config['target_padding_length'], wandb_plot=True)
        # visualize high vs low risk PT kinematics
        # plot_kinematics_mean_std(kinematics, pt_risk_indx, config['selected_opensim_labels'],
        #                          activity, config['target_padding_length'], wandb_plot=True)

        # run pca over kinematics and display relevant plots over kinematics
        pcanalysis_handler = PCAnalysis(config, kinematics, config['target_padding_length'], pc_variance=0.75)
        kinematics_pca, kinematic_pcs_comp, kinematic_pcs_loading, pca_variance_ratio, kinematic_pcs_transformed = pcanalysis_handler.apply_pca()
        if pca_save:
            pk.dump(kinematics_pca, open('./cache/v02/pca_' + activity + '.pkl', 'wb'))
        pcanalysis_handler.display_pca_variance_ratio(title=activity, wandb_plot=True)
        mode_variations = pcanalysis_handler.form_mode_variations(n_mode=len(kinematic_pcs_comp), std_factor=2)
        if activity == 'Gait':
            selected_PCs = [2, 3]
            selected_mode = ['pc2', 'pc3']
        elif activity == 'Stair Ascent':
            selected_PCs = [4]
            selected_mode = ['pc4']
        elif activity == 'Stair Descent':
            selected_PCs = [1]
            selected_mode = ['pc1']
        elif activity == 'STS':
            selected_PCs = [0, 1]
            selected_mode = ['pc1', 'pc2']
        else:
            selected_PCs = None
            selected_mode = ['pc1', 'pc2', 'pc3', 'pc4']

        # pcanalysis_handler.display_pcs_mode_profile(n_mode=len(kinematic_pcs_comp), std_factor=2, title=activity, selected_mode=selected_mode, wandb_plot=True)                         # plot pc mode profiles
        # pcanalysis_handler.display_range_of_each_pc_mode(title=activity, wandb_plot=True)                                                                  # plot range
        # pcanalysis_handler.display_pcs_comp_profile(title=activity, abs_pcs_status=False, wandb_plot=True, selected_PCs=None)                              # plot pcs component and pc abs component
        # pcanalysis_handler.display_pcs_loading_profile(title=activity + '_loading vector', abs_pcs_status=False,
        #                                                wandb_plot=True, selected_PCs=None)
        # pcanalysis_handler.display_pcs_loading_profile(title=activity + '_loading vector', abs_pcs_status=False, wandb_plot=True, selected_PCs=selected_PCs)
        # pcanalysis_handler.display_pcs_segment_profile(title=activity, wandb_plot=True)                                                 # plot pcs segment
        # pcanalysis_handler.display_4linkage_mechanism(n_mode=len(kinematic_pcs_comp), std_factor=4, title=activity, wandb_plot=True)    # plot 4 linkage mechanism
        # # pcanalysis_handler.display_4linkage_mechanism_animation(n_mode=len(kinematic_pcs_comp), std_factor=2, title=activity,
        # #                                               wandb_plot=True)  # plot 4 linkage mechanism

        # pcanalysis_handler.display_pcs_segment_profile_by_group(title=activity, group_index=tka_oa_indx, wandb_plot=True)
        # pcanalysis_handler.form_mode_variations_by_group(n_mode=len(kinematic_pcs_comp), std_factor=2, group_index=tka_oa_indx)
        # pcanalysis_handler.display_pcs_mode_profile_by_group(n_mode=len(kinematic_pcs_comp), std_factor=2, title=activity, group_index=tka_oa_indx, colors_groups=['b', 'r'], wandb_plot=True)
        # pcanalysis_handler.display_range_of_each_pc_mode_by_group(title=activity, group_name='knee', group_index=tka_oa_indx,
        #                                                           wandb_plot=True, pallete=['b', 'r'])

        # pcanalysis_handler.form_mode_variations_by_group(n_mode=len(kinematic_pcs_comp), std_factor=2, group_index=male_female_indx)
        # pcanalysis_handler.display_pcs_mode_profile_by_group(n_mode=len(kinematic_pcs_comp), std_factor=2, title=activity, group_index=male_female_indx, colors_groups=[plt.cm.Set2(i) for i in range(10)], wandb_plot=True)
        # pcanalysis_handler.display_range_of_each_pc_mode_by_group(title=activity, group_name='gender', group_index=male_female_indx, wandb_plot=True, pallete="Set2")

        # pd.DataFrame(original_vs_synthetic).to_excel('original_vs_synthetic.xlsx')
        if speed_status:
            pcanalysis_handler.form_mode_variations_by_group(n_mode=len(kinematic_pcs_comp), std_factor=2,
                                                             group_index=slow_normal_fast_indx)
            pcanalysis_handler.display_pcs_mode_profile_by_group(n_mode=len(kinematic_pcs_comp), std_factor=2,
                                                                 title=activity, group_index=slow_normal_fast_indx, colors_groups=[plt.cm.Set3(i) for i in range(10)],
                                                                 wandb_plot=True)
            pcanalysis_handler.display_range_of_each_pc_mode_by_group(title=activity, group_name='speed',
                                                                      group_index=slow_normal_fast_indx, wandb_plot=True,
                                                                      pallete="Set3")

        # form a dataframe from pcs and grouped_subject_speed
        pcs_columns = ['pc' + str(i + 1) for i in range(kinematic_pcs_transformed.shape[1])]
        grouped_subject_speed_pcs = pd.concat([grouped_subject_speed,
                                               pd.DataFrame(kinematic_pcs_transformed, columns=pcs_columns,
                                                            index=grouped_subject_speed.index)], axis=1)
        if speed_status:
            s = [grouped_subject_speed_pcs.index.values[s][0] for s in range(len(grouped_subject_speed_pcs))]
            grouped_subject_speed_pcs['subject'] = s
            grouped_subject_speed_pcs['subject'] = grouped_subject_speed_pcs['subject'].astype('category')
            p = [grouped_subject_speed_pcs.index.values[p][1] for p in range(len(grouped_subject_speed_pcs))]
            grouped_subject_speed_pcs['speed'] = p
            grouped_subject_speed_pcs['speed'] = grouped_subject_speed_pcs['speed'].astype('category')
        else:
            s = [grouped_subject_speed_pcs.index.values[s] for s in range(len(grouped_subject_speed_pcs))]
            grouped_subject_speed_pcs['subject'] = s
            grouped_subject_speed_pcs['subject'] = grouped_subject_speed_pcs['subject'].astype('category')

        grouped_subject_speed_pcs = grouped_subject_speed_pcs.reset_index(drop=True)

        # add knee status and gender status columns to the grouped_subject_speed_pcs
        for key, value in encoder_functions.items():
            decoded_y = []
            for i in grouped_subject_speed_pcs[key]:
                decoded_y.append(value.inverse_transform([int(i)])[0])
            column_name = key[:-7]
            grouped_subject_speed_pcs[column_name] = decoded_y
            grouped_subject_speed_pcs[column_name] = grouped_subject_speed_pcs[column_name].astype('category')

        plt.figure()
        sns.boxplot()
        # Export data for statistical analysis
        if export_csv:
            selected_headers = ['knee status', 'gender status', 'Age', 'Height (CM)', 'weight (KG)', 'KOOS', 'subject', 'speed', 'knee', 'gender']
            selected_headers.extend((pcs_columns))
            grouped_subject_speed_pcs[selected_headers].to_csv(activity + '_pca.csv')

        # Scatter: display pair plot of pc_transformed kinematic and patient demographics
        if scatter_pairplot:
            subject_variables = config["patient_variable_headers"].copy()
            subject_variables.extend(pcs_columns)
            # for subject_variable in [['speed'], ['knee'], ['subject'], ['gender']]:
            for subject_variable in [['knee'], ['gender']]:
                subject_variable.extend((subject_variables))
                plot_scatter_pair(grouped_subject_speed_pcs[subject_variable], subject_variable[0], title=activity, wandb_plot=True)
                plt.tight_layout()

        # Spearman and Pearson Correlation between patient variables and calculated kinematic pcs: Spearman's ρ and Pearson's r
        corr_columns = config["patient_demographic_headers"].copy()
        corr_columns.extend(pcs_columns)
        corr_r_df = grouped_subject_speed_pcs[corr_columns].corr(method='pearson')
        # corr_p_df = grouped_subject_speed_pcs[corr_columns].corr(method='spearman')

        # Heatmap: display correlation heatmap
        corr_r_df_lower = corr_r_df.where(np.tril(np.ones(corr_r_df.shape)).astype(np.bool))
        # plot_heatmap(corr_r_df_lower, title='pearson correlation:patient_variable_pcs', subtitle=activity, wandb_plot=False)
        # corr_p_df_lower = corr_r_df.where(np.tril(np.ones(corr_p_df.shape)).astype(np.bool))

        # PCA2: 2nd round of PCS between patients variables and calculated kinematic pcs
        subjectpcsanalysis_handler = SubjectPCsAnalysis(config, subject_demographic_variables=grouped_subject_speed_pcs[config["patient_variable_headers"]],
                                                        x=kinematics, range_len=config['target_padding_length'], pc_variance=0.75,
                                                        pc_subject_variance=0.95, standardize_method=None)
        _, _, _, subject_kinematic_pcs_transformed = subjectpcsanalysis_handler.apply_pca_subject()
        subjectpcsanalysis_handler.display_pca_variance_ratio(title=activity, wandb_plot=True)
        combined_pc1s_pc2s_df = subjectpcsanalysis_handler.combine_PC1s_PC2s_df()
        # plot_heatmap(combined_pc1s_pc2s_df, xlabel='Input Variables of PC2 (patients + PCs calculated from previous step (Kinematics Variable--> PCs)',
        #              ylabel='PC2s Component Values', title='pca: patient_variable_pcs', subtitle=activity, wandb_plot=False)

        # save all the PCs for each activities in a dict

        if speed_status:
            if activity == 'Gait' and status == False:
                pcs_activities_df_speed = pd.melt(grouped_subject_speed_pcs[['speed'] + pcs_columns], id_vars=['speed'])
                plot_box_plot_pcs_hue(pcs_activities_df_speed, x="value", y="variable", hue="speed", palette="Set3",
                                      orient='h',
                                      wandb_plot=True)
                pcs_activities = []
                for s, speed in enumerate(['fast', 'normal', 'slow']):
                    pcs_activities.append(
                        pd.DataFrame(kinematic_pcs_transformed[np.arange(s, len(kinematic_pcs_transformed), 3), :],
                                     columns=[activity + '_' + speed + '_' + i for i in pcs_columns]))
                status = True
            else:
                if status == False:
                    pcs_activities = []
                    pcs_activities.append(pd.DataFrame(kinematic_pcs_transformed, columns=[activity+'_'+i for i in pcs_columns]))
                    status = True
                else:
                    pcs_activities.append(pd.DataFrame(kinematic_pcs_transformed, columns=[activity + '_' + i for i in pcs_columns]))
                    status = True
        else:
            if status == False:
                pcs_activities = []
                pcs_activities.append(
                    pd.DataFrame(kinematic_pcs_transformed, columns=[activity + '_' + i for i in pcs_columns]))
                status = True
            else:
                pcs_activities.append(
                    pd.DataFrame(kinematic_pcs_transformed, columns=[activity + '_' + i for i in pcs_columns]))
                status = True
        pcs_activities_variance_ratio[activity] = pca_variance_ratio.T
        mode_variations_activities[activity] = mode_variations

    ############################################
    mode_variations_activities_dic = {}
    include_mean = False
    for k, v in mode_variations_activities.items():
        for j, (k2, v2) in enumerate(v.items()):
            if j%2==0:
                mean_mode = []
                mean_mode.append(v2)
            else:
                mean_mode.append(v2)
            mode_variations_activities_dic[k+'_'+k2] = v2
            if j%2==1 and include_mean:
                mode_variations_activities_dic[k + '_mode_'+ k2[-3:] + '_mean'] = np.array(mean_mode).mean(axis=0)

    # class : time series similarity analysis
    timeseries_similarity_handler = TimeSeriesSimilarity(mode_variations_activities_dic, config)
    # plot means
    # timeseries_similarity_handler.plot_profiles('Gait', 'Stair')
    # correlation
    # timeseries_similarity_handler.plot_correlation_heatmap(correlation_method='pearson')
    # similarity measures , dtw, eluc, corr
    similarity_measures_df = timeseries_similarity_handler.calculate_similarity_measures(correlation_method='pearson', all_dof_together=True)
    corr = timeseries_similarity_handler.form_matrix_of_similarity_measures(measure='corr_r')
    ed = timeseries_similarity_handler.form_matrix_of_similarity_measures(measure='ed')
    dtw = timeseries_similarity_handler.form_matrix_of_similarity_measures(measure='dtw')
    corr_ave = timeseries_similarity_handler.form_matrix_of_similarity_measures_ave(data_df=corr)
    ed_ave = timeseries_similarity_handler.form_matrix_of_similarity_measures_ave(data_df=ed)
    dtw_ave = timeseries_similarity_handler.form_matrix_of_similarity_measures_ave(data_df=dtw)
    # plot_heatmap(abs(corr_ave.round(2)), title='corr r: all activities pcs', subtitle='',
    #              wandb_plot=False)
    # plot_heatmap(abs(ed_ave.round(1)), title='ed distance: all activities pcs', subtitle='',
    #              wandb_plot=False)
    # plot_heatmap(abs(dtw_ave.round(1)), title='dtw distance: all activities pcs', subtitle='',
    #              wandb_plot=False)

    for metric in ['dtw', 'ed', 'corr_r']:
        if metric != 'corr_r':
            similarity_measures_df[metric + '_norm'] = 1 / (
                        similarity_measures_df[metric] / similarity_measures_df[metric].min())

    # dtw similarity
    # timeseries_similarity_handler.dtw_plot(query1='Gait_mode + 2stdev_pc1', query2='Stair Ascent_mode + 2stdev_pc1', all_dof_together=True)
    # dtw clustering
    # timeseries_similarity_handler.dtw_clustering_plot(cluster_method='linkage', query1='Gait_mode + 2stdev_pc1', query2='Stair Ascent', all_dof_together=True)

    # form variance ratio df of all pcs
    min_length = max([len(value) for key, value in pcs_activities_variance_ratio.items()])
    pcs_activities_variance_ratio_df = pd.DataFrame({k: pd.Series(v[:min_length]) for k, v in pcs_activities_variance_ratio.items()})
    # form df of kinematic profile of all modes
    pcs_activities_df = pd.concat(pcs_activities, axis=1)
    # plot_heatmap(pcs_activities_df, title='All PCs', subtitle='', wandb_plot=False)

    p_variable = config['patient_variable_headers'].copy()
    p_variable.extend(['subject', 'gender', 'knee'])
    pcs_activities_patient_variables_df = pd.concat(
        [grouped_subject_speed_pcs[p_variable], pcs_activities_df], axis=1)
    oa = pcs_activities_patient_variables_df[pcs_activities_patient_variables_df['knee'] =='OA']
    tka = pcs_activities_patient_variables_df[pcs_activities_patient_variables_df['knee'] =='BiTKA']
    m = pcs_activities_patient_variables_df[pcs_activities_patient_variables_df['gender'] =='M']
    f = pcs_activities_patient_variables_df[pcs_activities_patient_variables_df['gender'] =='F']
    # pcs_activities_patient_variables_df.to_excel('all_pca_patient_variables.xlsx', sheet_name='all')

    if export_xlsx:
        with pd.ExcelWriter('all_pca_patient_variables_singlegait_v2.xlsx') as writer:
            pcs_activities_patient_variables_df.to_excel(writer, sheet_name='all')
            oa.to_excel(writer, sheet_name='oa')
            tka.to_excel(writer, sheet_name='tka')
            m.to_excel(writer, sheet_name='male')
            f.to_excel(writer, sheet_name='female')
            pcs_activities_variance_ratio_df.to_excel(writer, sheet_name='pcs_variance_ratio')
            similarity_measures_df.to_excel(writer, sheet_name='pcs_similarity_measures')

    # display PCs for knee, genders, subject
    pcs_activities_df_knee = pcs_activities_df.copy()
    pcs_activities_df_knee['knee'] = grouped_subject_speed_pcs['knee']
    pcs_activities_df_knee = pd.melt(pcs_activities_df_knee.dropna(), id_vars=['knee'])
    my_pal = {knee: plt.cm.Set1.colors[0] if knee == "OA" else plt.cm.Set1.colors[1] for knee in
              pcs_activities_df_knee.knee.unique()}
    plot_box_plot_pcs_hue(pcs_activities_df_knee, x="value", y="variable", hue="knee", palette=my_pal, orient='h',
                          wandb_plot=True)

    # box plot over PCs for gender
    pcs_activities_df_gender = pcs_activities_df.copy()
    pcs_activities_df_gender['gender'] = grouped_subject_speed_pcs['gender']
    pcs_activities_df_gender = pd.melt(pcs_activities_df_gender.dropna(), id_vars=['gender'])
    # plot_box_plot_pcs_hue(pcs_activities_df_gender, x="value", y="variable", hue="gender", palette="Set2", orient='h',
    #                       wandb_plot=True)

    if scatter_pairplot:
        pcs_activities_patient_knee_df = pd.concat(
            [grouped_subject_speed_pcs['knee'], pcs_activities_df], axis=1)
        plot_scatter_pair(pcs_activities_patient_knee_df, 'knee', title=activity,
                              wandb_plot=True)
        sns.pairplot(pcs_activities_df, kind='reg', corner=True)

    # clustering and correlation between activities PCs
    r = 'r2'
    corr_r_pcs_activities_df = abs(pcs_activities_df.corr(method='pearson'))
    # corr_r_pcs_activities_df = pcs_activities_df.corr(method='pearson')
    corr_r_pcs_activities_df_lower = corr_r_pcs_activities_df.where(np.tril(np.ones(corr_r_pcs_activities_df.shape)).astype(np.bool))
    corr_r_pcs_activities_df_lower = corr_r_pcs_activities_df_lower.iloc[np.arange(4,14)][corr_r_pcs_activities_df_lower.columns[~corr_r_pcs_activities_df_lower.columns.str.contains('Descent')]].round(5)
    corr_r_pcs_activities_df_lower.replace(0, np.nan, inplace=True)
    corr_r_pcs_activities_df_lower.replace(1, np.nan, inplace=True)
    r = 'r2'
    if r=='r2':
        corr_df = corr_r_pcs_activities_df_lower.round(2)**2
        corr_df = corr_df.round(2)
        title = 'heatmap_actvities_r2'
    elif r == 'abs':
        corr_df = abs(corr_r_pcs_activities_df_lower.round(2))
        title = 'heatmap_actvities_abs'
    else:
        corr_df = corr_r_pcs_activities_df_lower.round(2)
        title = 'heatmap_actvities'
    plot_heatmap(corr_df, title=title,
                     subtitle='', wandb_plot=False)


    # plot_cluster_heatmap(pcs_activities_df, metric='correlation',
    #                      title='pearson correlation: all activities pcs', wandb_plot=False)
    # plot_cluster_heatmap(corr_r_pcs_activities_df, method='single', metric='euclidean',
    #                      title='pearson correlation: all activities pcs euclidean', wandb_plot=False)

    # correlation graph network:
    corr_method = 'pearson'
    plot_graph_network(pcs_activities_df, metric='correlation', corr_method=corr_method, lower_bound_threshold=0.6, upper_bound_threshold=0.9,
                       title='title', wandb_plot=True)

    # patient variables and all pcs relationship
    pcs_activities_std_df = pcs_activities_df.copy()
    for activity in config['risk_factor_activity'][0:3]:
        pcs_activities_df_temp = pcs_activities_df[[col for col in pcs_activities_df.columns if activity in col]].copy()
        scaler_pcs = StandardScaler()
        scaler_fit_pcs = scaler_pcs.fit(pcs_activities_df_temp.values.reshape(-1, 1))
        x_transformed_scaled = scaler_fit_pcs.transform(pcs_activities_df_temp.values.reshape(-1, 1))
        x_transformed_scaled = np.reshape(x_transformed_scaled,
                                          [pcs_activities_df_temp.values.shape[0],
                                           pcs_activities_df_temp.values.shape[1]], 'F')
        pcs_activities_std_df[[col for col in pcs_activities_df.columns if activity in col]] = x_transformed_scaled

    config['patient_demographic_headers'] = ["Age", "Height (CM)", "weight (KG)", "KOOS"]
    numeric_subject_variable_std_df = grouped_subject_speed_pcs[config['patient_demographic_headers']].copy()
    for patient_variable in config['patient_demographic_headers']:
        numeric_subject_variable_df_temp = numeric_subject_variable_std_df[patient_variable].copy()
        scaler_pcs = StandardScaler()
        scaler_fit_pcs = scaler_pcs.fit(numeric_subject_variable_df_temp.values.reshape(-1, 1))
        x_transformed_scaled = scaler_fit_pcs.transform(numeric_subject_variable_df_temp.values.reshape(-1, 1))
        numeric_subject_variable_std_df[patient_variable] = x_transformed_scaled

    # clustem heatmap: standardize-> patient variables and all pcs, std
    corr_r_subject_pcs_activities_std_df = pd.concat([numeric_subject_variable_std_df, pcs_activities_std_df], axis=1).corr(method='pearson')
    # plot_cluster_heatmap(corr_r_subject_pcs_activities_std_df, method='single', metric='euclidean',
    #                      title='pearson correlation: patient variables and all activities pcs euclidean', wandb_plot=False)
    corr_r_subject_pcs_activities_df_lower = corr_r_subject_pcs_activities_std_df.where(
        np.triu(np.ones(corr_r_subject_pcs_activities_std_df.shape)).astype(np.bool))
    # filter
    corr_r_subject_pcs_activities_df_lower = corr_r_subject_pcs_activities_df_lower.iloc[np.arange(0,4)][pcs_activities_std_df.columns]
    r = 'r2'
    if r=='r2':
        corr_df = corr_r_subject_pcs_activities_df_lower.round(2)**2
        corr_df = corr_df.round(2)
        title = 'heatmap_activities_demographics_r2'
    else:
        corr_df = corr_r_subject_pcs_activities_df_lower.round(2)
        title = 'heatmap_activities_demographics'
    plot_heatmap_demographic(corr_df, title=title,
                 subtitle='', wandb_plot=False)


    # cluster heatmap: non standarize -> patient variables and all pcs
    corr_method = 'pearson'
    weight = grouped_subject_speed_pcs[config['patient_demographic_headers']]['weight (KG)']
    height = grouped_subject_speed_pcs[config['patient_demographic_headers']]['Height (CM)']/100
    grouped_subject_speed_pcs['BMI'] = weight/height/height
    ant = ['BMI'].append(config['patient_demographic_headers'][0:4])
    corr_r_subject_pcs_activities_df = pd.concat([grouped_subject_speed_pcs[['BMI'] + config['patient_demographic_headers'][0:4]], pcs_activities_df], axis=1).corr(method=corr_method)
    corr_r_subject_pcs_activities_df_lower = corr_r_subject_pcs_activities_df.where(
        np.triu(np.ones(corr_r_subject_pcs_activities_df.shape)).astype(np.bool))
    # filter
    corr_r_subject_pcs_activities_df_lower = corr_r_subject_pcs_activities_df_lower.iloc[np.arange(0,5)][pcs_activities_df.columns]
    r = 'abs(r)'
    if r=='r2':
        corr_df = corr_r_subject_pcs_activities_df_lower.round(2)**2
        corr_df = corr_df.round(2)
        title = 'heatmap_activities_demographics_r2'
    elif r == 'abs(r)':
        corr_df = abs(corr_r_subject_pcs_activities_df_lower.round(2))
        corr_df = corr_df.round(2)
        title = 'heatmap_activities_demographics_r_abs'
    else:
        corr_df = corr_r_subject_pcs_activities_df_lower.round(2)
        title = 'heatmap_activities_demographics'
    plot_heatmap_demographic(corr_df, title=title,
                 subtitle='', wandb_plot=False)

    plot_heatmap(corr_r_subject_pcs_activities_df_lower.round(2), title=corr_method+ ' correlation: patient variable & all activities pcs',
                 subtitle='', wandb_plot=False)
    # plot_cluster_heatmap(corr_r_subject_pcs_activities_df, method='single', metric='correlation',
    #                      title=corr_method + ' correlation: patient variables and all activities pcs euclidean', wandb_plot=False)
    plot_graph_network(pcs_activities_df,  grouped_subject_speed_pcs[config['patient_demographic_headers']], metric='correlation', corr_method=corr_method, lower_bound_threshold=0.4, upper_bound_threshold=1,
                       title='title', wandb_plot=True)
    plot_graph_network(corr_r_subject_pcs_activities_df_lower, metric='correlation', corr_method=corr_method, lower_bound_threshold=0.4, upper_bound_threshold=1,
                       title='title', wandb_plot=True)

if __name__ == '__main__':
    run_main()