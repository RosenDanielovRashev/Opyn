import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

st.title("Изолиния Ei/Ed = 2 (H/D по x, y по y) с интерполация")

df = pd.read_csv('danni.csv')

h_d_unique = np.linspace(df['H/D'].min(), df['H/D'].max(), 100)
y_unique = np.linspace(df['y'].min(), df['y'].max(), 100)

grid_h, grid_y = np.meshgrid(h_d_unique, y_unique)

points = df[['H/D', 'y']].values
values = df['Ei/Ed'].values

grid_z = griddata(points, values, (grid_h, grid_y), method='cubic')

fig = go.Figure(data=go.Contour(
    z=grid_z,
    x=h_d_unique,
    y=y_unique,
    contours=dict(
        coloring='lines',
        showlabels=True,
        start=2,
        end=2,
        size=1
    ),
    line=dict(width=3, color='red')
))

fig.update_layout(
    xaxis_title='H/D',
    yaxis_title='y',
    xaxis=dict(
        dtick=0.1  # деления през 0.1 по x
    )
)

st.plotly_chart(fig)
