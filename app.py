import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Точки с Ei/Ed близко до 2 (без интерполация)")

df = pd.read_csv('danni.csv')

# Филтрираме точки, където Ei/Ed е близо до 2 (+- 0.05 примерно)
tol = 0.05
df_2 = df[(df['Ei/Ed'] >= 2 - tol) & (df['Ei/Ed'] <= 2 + tol)]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_2['H/D'],
    y=df_2['y'],
    mode='markers',
    marker=dict(color='red', size=6),
    name='Ei/Ed ≈ 2'
))

fig.update_layout(
    xaxis_title='H/D',
    yaxis_title='y',
    title='Точки с Ei/Ed около 2'
)

st.plotly_chart(fig)
