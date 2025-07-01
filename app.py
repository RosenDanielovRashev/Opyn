import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

st.title("Изолиния Ei/Ed = 2 (H/D по x с гъстота 0.1, y по y) с интерполация")

df = pd.read_csv('danni.csv')

# Гъстота по H/D с крачка 0.1
h_d_unique = np.arange(df['H/D'].min(), df['H/D'].max() + 0.1, 0.1)

# 100 точки по y
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
)

st.plotly_chart(fig)
