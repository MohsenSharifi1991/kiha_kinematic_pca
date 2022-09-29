import streamlit as st
import numpy as np
import pandas as pd



st.title('real time data manupilation')
def draw_y(alpha, beta):
    x = np.arange(0, 100, 0.5)
    y = np.sin(alpha*x+beta)
    return x, y

alpha = st.slider('alphe', float(-5), float(5), step=0.1)
beta = st.slider('beta', 0, 5)
x, y = draw_y(alpha, beta)
# st.line_chart(x, y)
dics = {'y':y}
chart_data = pd.DataFrame(dics)
# chart_data = pd.DataFrame(
#      np.random.randn(20, 2),
#      columns=['x', 'y'])

st.line_chart(chart_data)

