import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Изолинии на Ei/Ed по y и H/D")

# Зареждане на CSV (слагай правилния път)
df = pd.read_csv('danni.csv')

# Проверка на размерите
st.write(f"Брой редове: {len(df)}")

# Създаване на grid
y_unique = np.sort(df['y'].unique())
h_d_unique = np.sort(df['H/D'].unique())
Y, H_D = np.meshgrid(y_unique, h_d_unique)

Z = np.full(Y.shape, np.nan)

for i, h in enumerate(h_d_unique):
    for j, y_val in enumerate(y_unique):
        val = df[(df['y'] == y_val) & (df['H/D'] == h)]['Ei/Ed']
        if not val.empty:
            Z[i, j] = val.values[0]

fig = go.Figure(data=go.Contour(
    z=Z,
    x=y_unique,
    y=h_d_unique,
    colorscale='Viridis',
    contours=dict(coloring='lines', showlabels=True)
))

st.plotly_chart(fig)

