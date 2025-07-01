import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Примерни данни (замени с твоя csv)
y_unique = np.linspace(0, 10, 20)
h_d_unique = np.linspace(0, 5, 10)
Y, H_D = np.meshgrid(y_unique, h_d_unique)
Z = np.sin(Y) * np.cos(H_D)  # примерно Ei/Ed

fig = go.Figure(data=go.Contour(
    z=Z,
    x=y_unique,
    y=h_d_unique,
    colorscale='Viridis',
    contours=dict(coloring='lines', showlabels=True)
))

st.plotly_chart(fig)
