import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy

from visualization.matplotly_plot import plot_heatmap, plot_warping_kiha, plot_cluster_dtw
import pandas as pd
from tslearn.metrics import dtw as tslearndtw
from dtaidistance import clustering
from dtaidistance import dtw
from dtaidistance import ed

class TimeSeriesSimilarity:
    def __init__(self, timeseries_dic, config):
        self.data_dic = timeseries_dic
        self.data_df = pd.DataFrame.from_dict(timeseries_dic)
        self.config = config

    def plot_profiles(self, query1, query2):
        plt.figure(figsize=(10, 8))
        n_kinematic = len(self.config['selected_opensim_labels'])
        c = 2
        range_len = 100
        for key, value in self.data_dic.items():
            if query1 in key or query2 in key:
                for i, kinematic_label in enumerate(self.config['selected_opensim_labels']):
                    plt.subplot(round(n_kinematic / c), c, i + 1)
                    len_data = range(i * range_len, i * range_len + range_len)
                    plt.plot(value[len_data], label=key, linestyle='-')
        plt.legend()
        plt.show()

    def plot_correlation_heatmap(self, correlation_method):
        corr_r_mode_variations_activities_df = self.data_df.corr(method=correlation_method)
        corr_r_mode_variations_activities_df_lower = corr_r_mode_variations_activities_df.where(
            np.tril(np.ones(corr_r_mode_variations_activities_df.shape)).astype(np.bool))
        plot_heatmap(corr_r_mode_variations_activities_df_lower.round(1), title='pearson correlation: all modes', subtitle='',
                     wandb_plot=False)
        return corr_r_mode_variations_activities_df_lower

    def calculate_similarity_measures(self, correlation_method='pearson', all_dof_together=True):
        vertices = self.data_df.columns.values.tolist()
        range_len = self.config['target_padding_length']
        us, vs, dofs = [], [], []
        dtw_distance, ed_distance, corr_r = [], [], []
        for u, v in itertools.combinations(vertices, 2):
            s1 = self.data_df[u].values
            s2 = self.data_df[v].values
            if all_dof_together:
                # dtw_distance.append(dtw.distance(s1, s2))
                dtw_distance.append(dtw.distance_fast(s1, s2, use_pruning=True))
                ed_distance.append(ed.distance(s1, s2))
                corr_r.append(scipy.stats.pearsonr(s1, s2)[0])
                us.append(u)
                vs.append(v)
                dofs.append([])
            else:
                for i, kinematic_label in enumerate(self.config['selected_opensim_labels']):
                    len_data = range(i * range_len, i * range_len + range_len)
                    # dtw_distance.append(dtw.distance(s1[len_data], s2[len_data]))
                    dtw_distance.append(dtw.distance_fast(s1[len_data], s2[len_data], use_pruning=True))
                    ed_distance.append(ed.distance(s1[len_data], s2[len_data]))
                    corr_r.append(scipy.stats.pearsonr(s1[len_data], s2[len_data])[0])
                    us.append(u)
                    vs.append(v)
                    dofs.append(kinematic_label)
        self.similarity_measures = pd.DataFrame.from_dict({'us': us, 'vs':vs, 'labels':dofs, 'dtw':dtw_distance, 'ed':ed_distance, 'corr_r':corr_r})
        return self.similarity_measures

    def form_matrix_of_similarity_measures(self, measure):
        if measure=='corr_r':
            best_measure=1
        elif measure == 'dtw' or measure=='ed':
            best_measure = 0

        metric = []
        for i, pc in enumerate(self.similarity_measures['us'].unique()):
            metric.append(np.hstack((best_measure, self.similarity_measures[measure][self.similarity_measures['us'] == pc].values)).T)
        metric.append(np.array([best_measure]))
        b = np.zeros([len(metric), len(max(metric, key=lambda x: len(x)))])
        b[:] = np.NaN
        for i, j in enumerate(metric):
            b[len(metric) - len(j):, i] = j
        similarity_measures_df = pd.DataFrame(data=b, index=list(self.data_dic.keys()),
                                                  columns=list(self.data_dic.keys()))
        return similarity_measures_df


    def form_matrix_of_similarity_measures_ave(self, data_df):
        data_df_ave = np.zeros([int(len(data_df)/2), int(len(data_df)/2)])
        data_df_ave[:] = np.NaN
        for i in range(0, int(len(data_df)/2)):
            for j in range(0, int(len(data_df)/2)):
                data_df_ave[i, j] = data_df.iloc[i * 2:i * 2 + 2, j * 2:j * 2 + 2].mean().mean()

        index = [key.replace('+', '') for key in list(data_df.columns.values) if '+' in key]
        data_df_ave = pd.DataFrame(data=data_df_ave, index=index, columns=index)
        return data_df_ave

    def dtw_plot(self, query1, query2, all_dof_together=True):
        vertices = self.data_df.columns.values.tolist()
        for u, v in itertools.combinations(vertices, 2):
            if query1 in u and query2 in v:
                s1 = self.data_df[u].values
                s2 = self.data_df[v].values
                plot_warping_kiha(s1, s2, self.config['selected_opensim_labels'], self.config['target_padding_length'],
                                  title='\n' + u + '\n' + v, all_dof_together=all_dof_together, filename=None)


    def dtw_clustering_plot(self, cluster_method, query1, query2, all_dof_together=True):
        column_name = [c for c in self.data_df.columns if query1 in c or query2 in c]
        timeseries_df = self.data_df[column_name]
        plot_cluster_dtw(timeseries_df, cluster_method, self.config['selected_opensim_labels'],
                         self.config['target_padding_length'], 'all dof', all_dof_together=all_dof_together, filename=None)

