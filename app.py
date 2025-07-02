import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr")

# Вход: брой пластове
n = st.number_input("Брой пластове (n)", min_value=1, step=1, value=3)

# Създаване на списъци за h_i и E_i
h_values = []
E_values = []

st.markdown("### Въведи стойности за всеки пласт")

cols = st.columns(2)
for i in range(n):
    with cols[0]:
        h = st.number_input(f"h_{i+1}", value=1.0, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        E = st.number_input(f"E_{i+1}", value=1.0, step=0.1, key=f"E_{i}")
        E_values.append(E)

# Изчисляване на Esr и H
h_array = np.array(h_values)
E_array = np.array(E_values)

sum_h = h_array.sum()
Esr = np.sum(E_array * h_array) / sum_h if sum_h != 0 else 0

# Показване на формулите с LaTeX
st.latex(r"Esr = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}")
st.latex(r"H = \sum_{i=1}^{n} h_i")

# Показване на резултатите
st.write(f"Обща дебелина H = {sum_h:.3f}")
st.write(f"Изчислено Esr = {Esr:.3f}")

# Тук продължава твоя код с графиката...

# Зареждане на оригиналните данни
df_original = pd.read_csv("danni.csv")

# Зареждане на новите данни
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

fig = go.Figure()

# Добавяне на оригиналните изолинии за Ei/Ed
if 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {level}',
            line=dict(width=2)
        ))

# Добавяне на новите изолинии за sr_Ei
if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'σsr/Ei = {sr_Ei}',
            line=dict(width=2)
        ))

# Празна следа за горната ос (xaxis2)
fig.add_trace(go.Scatter(
    x=[0,1],
    y=[None,None],
    mode='lines',
    xaxis='x2',
    showlegend=False,
    hoverinfo='skip'
))

fig.update_layout(
    width=700,
    height=700,
    xaxis=dict(
        title='H/D',
        dtick=0.1,
        range=[0, 2],
        constrain='domain',
        domain=[0, 1]
    ),
    xaxis2=dict(
        title='σₙ',
        overlaying='x',
        side='top',
        range=[0, 1],
        tickmode='linear',
        tick0=0,
        dtick=0.1,
        showgrid=False
    ),
    yaxis=dict(
        title='y',
        dtick=0.1,
        range=[0, 2.5],
        scaleanchor='x',
        scaleratio=1
    ),
    title='Комбинирани изолинии',
    legend=dict(title='Легенда')
)

st.plotly_chart(fig, use_container_width=False)
