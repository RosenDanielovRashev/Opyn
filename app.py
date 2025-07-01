import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata

st.title("Изолиния Ei/Ed = 2 с интерполация")

# Зареждане на CSV файла
df = pd.read_csv('danni.csv')

# Създаваме гъста мрежа за интерполация
y_unique = np.linspace(df['y'].min(), df['y'].max(), 100)
h_d_unique = np.linspace(df['H/D'].min(), df['H/D'].max(), 100)
grid_y, grid_h = np.meshgrid(y_unique, h_d_unique)

# Вземаме реалните точки и стойности
points = df[['y', 'H/D']].values
values = df['Ei/Ed'].values

# Интерполираме стойностите върху мрежата
grid_z = griddata(points, values, (grid_y, grid_h), method='cubic')

# Създаваме контурна графика с ЕДИН контур за стойност 2
fig = go.Figure(data=go.Contour(
    z=grid_z,
    x=y_unique,
    y=h_d_unique,
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
    xaxis_title='y',
    yaxis_title='H/D',
)

st.plotly_chart(fig)
