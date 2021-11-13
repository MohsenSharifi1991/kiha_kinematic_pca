import pickle as pk
import numpy as np
from config import get_config
from visualization.matplotly_plot import plot_kinematics
from visualization.streamlit_graph import plot_kinematic

config = get_config()
# load pca models, scaler, activity density, and store them in dict
activity_densities = pk.load(open('./cache/activity_densities.pkl', 'rb'))
generative_model = {'Gait': {}, 'Stair Ascent': {}, 'Stair Descent': {}, 'STS': {}}
for key, value in generative_model.items():
    # load scalar
    value['pca'] = pk.load(open('./cache/pca_'+key+'.pkl', 'rb'))
    value['scalar'] = pk.load(open('./cache/pcs_scaler_' + key + '.pkl', 'rb'))
    value['density_function'] = activity_densities[key]


def run_generative_model(selected_activity, selected_knee, number_sample):
    pca = generative_model[selected_activity]['pca']
    scaler = generative_model[selected_activity]['scalar']
    df = generative_model[selected_activity]['density_function']
    mu = [v for k, v in df.items() if selected_knee in k and 'mu' in k][0]
    cov = [v for k, v in df.items() if selected_knee in k and 'cov' in k][0]
    x_standardized = np.random.multivariate_normal(mu, cov, number_sample)
    x_re_standardized = scaler.inverse_transform(x_standardized)
    x = pca.inverse_transform(x_re_standardized)
    return x


def run_prediction_pcs_kinematics(input_data, selected_input_activity, selected_output_activities, selected_model):
    x_list = []
    prediction_output = {}
    for output_activity in selected_output_activities:
        '''
        # 1) new input data--> PCA --> PCs
        # 2) PCs --> scalar --> std PCs
        # 3) std PCs --> regression --> std PCs of new activity
        # 4) std PCs of new activity --> re-scalar --> PCs of new activity # not necessary
        # 5) PCs of new activity--> re PCA --> new output data
        '''
        pca_input = generative_model[selected_input_activity]['pca']
        pca_output = generative_model[output_activity]['pca']
        pipeline = pk.load(open('./cache/regression_models/model_' + selected_model + '_' + selected_input_activity
        + '_' + output_activity + '.pkl', 'rb'))
        pcs = pca_input.transform(input_data)
        new_pcs = pipeline.predict(pcs)
        new_x = pca_output.inverse_transform(new_pcs)
        x_list.append(new_x)
    for key, value in zip(selected_output_activities, x_list):
        prediction_output[key] = value
    return prediction_output


if __name__ == '__main__':
    selected_activity = 'Stair Ascent'
    selected_knee = 'oa'
    input_data = np.random.randint(100, size=(3,800))
    selected_input_activity = 'Stair Ascent'
    selected_output_activities = ['Gait', 'STS']
    selected_model = 'LinearRegression'
    new_y = run_prediction_pcs_kinematics(input_data, selected_input_activity, selected_output_activities, selected_model)
    new_x = run_generative_model(selected_activity=selected_activity, selected_knee='oa', number_sample=5)
    # plot_kinematics(new_x, config['selected_opensim_labels'], selected_activity + '_'+selected_knee, config['target_padding_length'], False)
    fig = plot_kinematic(new_x, config['selected_opensim_labels'], selected_activity + '_' + selected_knee,
                    config['target_padding_length'])
    a = 1