import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Линия на Ei/Ed ≈ 2 без интерполация")

df = pd.read_csv('danni.csv')

# Филтрираме точки близо до 2 (толеранс +-0.05)
tol = 0.05
df_2 = df[(df['Ei/Ed'] >= 2 - tol) & (df['Ei/Ed'] <= 2 + tol)]

# Сортираме по H/D (или y) за свързване с линия
df_2 = df_2.sort_values(by=['H/D', 'y'])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_2['H/D'],
    y=df_2['y'],
    mode='lines+markers',  # линии + точки
    line=dict(color='red', width=2),
    marker=dict(size=6, color='red'),
    name='Ei/Ed ≈ 2'
))

fig.update_layout(
    xaxis_title='H/D',
    yaxis_title='y',
    title='Свързана линия на точки с Ei/Ed около 2',
    showlegend=True
)

st.plotly_chart(fig)
