import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot_kinematic(data, dof_labels, activity, range_len):
    colors = px.colors.sequential.Turbo
    colors2 = px.colors.sequential.Plasma
    colors.extend(colors2)
    # colors = list(range(-3, 10))
    n_kinematic = len(dof_labels)
    fig = make_subplots(rows=n_kinematic, cols=2)
    for i, kinematic_label in enumerate(dof_labels):
        len_data = range(i * range_len, i * range_len + range_len)
        for j, y in enumerate(data):
            fig.append_trace(go.Scatter(x=np.arange(0, len(len_data)), y=y[len_data],
                                        mode='lines', line_color=colors[j], name=''), row=(i // 2) + 1, col=(i % 2) + 1)

            fig.update_yaxes(title_text='cycle', row=(i // 2) + 1, col=(i % 2) + 1)
            fig.update_yaxes(title_text=kinematic_label + '(deg)', row=(i // 2) + 1, col=(i % 2) + 1)
    fig.update_layout(height=1500, width=800,showlegend=False,
                      title=activity)
    # fig.update_layout(showlegend=False,
    #                   title=activity)
    # fig.layout.plot_bgcolor = app_color["graph_bg"]
    # fig.layout.paper_bgcolor = app_color["graph_bg"]
    return fig