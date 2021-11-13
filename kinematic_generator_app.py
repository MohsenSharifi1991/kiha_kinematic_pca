import streamlit as st

from config import get_config
from generator import run_generative_model, run_prediction_pcs_kinematics
from visualization.streamlit_graph import plot_kinematic
import pandas as pd
import base64
from io import BytesIO

config = get_config()


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>' # decode b'abc' => abc


# @st.cache  # 👈 This function will be cached
def load_data(uploaded_data_file):
    if uploaded_data_file is not None:
        file_details = {"Filename": uploaded_data_file.name, "FileType": uploaded_data_file.type,
                        "FileSize": uploaded_data_file.size}
        # st.write(file_details)
        if file_details["Filename"][-3:] == 'csv':
            df = pd.read_csv(uploaded_data_file, index_col=False)
        else:
            df = pd.read_excel(uploaded_data_file, index_col=False)

        if len(df.values.shape) == 2 and (df.values.shape[0]==800 or df.values.shape[1]==800):
            st.write('Uploaded file is in correct format')
            if df.values.shape[0]==800:
                uploaded_datas = df.values.T
            else:
                uploaded_datas = df.values
            return uploaded_datas
        else:
            return None




st.title('Synthetic Kinematic Generator')
st.sidebar.title('Select Task')
selected_app = st.sidebar.selectbox('Pick one', ['activity to activity', 'new kinematic'])

if selected_app == 'activity to activity':
    correct_file_format = False
    uploaded_data_file = st.sidebar.file_uploader("Upload CSV or EXCEL", type=['csv', 'xlsx'])

    st.sidebar.title("Select Input Activity")
    selected_input_activity = st.sidebar.selectbox('Pick one', ['Gait', 'Stair Ascent', 'Stair Descent', 'STS'])

    st.sidebar.title("Select Output Activity")
    selected_output_activities = st.sidebar.multiselect('Pick multiple', ['Gait', 'Stair Ascent', 'Stair Descent', 'STS'])

    st.sidebar.title("Select Model")
    selected_model = st.sidebar.selectbox('Pick one',
                                                      ['LinearRegression',
                                                       'RidgeCV',
                                                        'Lasso',
                                                        'ElasticNet',
                                                        'BayesianRidge',
                                                        'DecisionTreeRegressor',
                                                        'SVMLinear',
                                                        'SVMPoly',
                                                        'SVMRbf',
                                                        'KNeighborsRegressor'])

    if st.sidebar.button('Generate'):
        df = load_data(uploaded_data_file)
        if df is None:
            st.error('Upload a correct file format to [n, 800] or [800, n]')
        else:
            st.subheader('original data')
            fig = plot_kinematic(df, config['selected_opensim_labels'], selected_input_activity,
                                 config['target_padding_length'])
            st.dataframe(df)
            st.dataframe(df)
            c = st.columns(2)
            for i, kinematic_label in enumerate(config['selected_opensim_labels']):
                len_data = range(i * 100, i * 100 + 100)
                c[(i%2)].write(kinematic_label)
                c[(i%2)].line_chart(df.T[len_data])
            st.plotly_chart(fig)

            new_xs_all = {}
            new_xs = run_prediction_pcs_kinematics(df, selected_input_activity, selected_output_activities,
                                                   selected_model)
            for a, (activity, new_x) in enumerate(new_xs.items()):
                for s in range(len(new_x)):
                    new_xs_all[activity+'_'+str(s)] = new_x[s]
                fig = plot_kinematic(new_x, config['selected_opensim_labels'], activity,
                                     config['target_padding_length'])
                st.plotly_chart(fig)
            st.sidebar.title('Link of Generated Data')
            new_xs_all_df = pd.DataFrame.from_dict(new_xs_all)
            st.dataframe(new_xs_all_df)
            st.sidebar.markdown(get_table_download_link(new_xs_all_df), unsafe_allow_html=True)


else:
    st.sidebar.title("Select Activity")
    selected_activity = st.sidebar.selectbox('Pick one', ['Gait', 'Stair Ascent', 'Stair Descent', 'STS'])

    st.sidebar.title("Select Knee")
    selected_knee = st.sidebar.selectbox('Pick one', ['OA', 'TKA'])
    if selected_knee == 'OA':
        selected_knee = 'oa'
    else:
        selected_knee = 'tka'

    st.sidebar.title("Enter Number of Sample")
    n_sample = st.sidebar.number_input('Enter a Number Between 2 to 1000', 2, 1000)

    if st.sidebar.button('Generate'):
        new_x = run_generative_model(selected_activity=selected_activity, selected_knee=selected_knee, number_sample=n_sample)
        print(1//2)
        print(1%2)
        fig = plot_kinematic(new_x, config['selected_opensim_labels'], selected_activity + '_' + selected_knee,
                        config['target_padding_length'])
        st.plotly_chart(fig)

        st.sidebar.title('Link of Generated Data')
        df = pd.DataFrame(new_x.T, columns=['s_' + str(s) for s in range(n_sample)])
        st.dataframe(df)
        st.sidebar.markdown(get_table_download_link(df), unsafe_allow_html=True)




