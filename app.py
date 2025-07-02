import streamlit as st 
import pandas as pd
import plotly.graph_objects as go

st.title("Изолинии с две хоризонтални оси")

# Зареждане на данни
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sigma_r'}, inplace=True)  # Променено име на колоната

# Създаване на фигура
fig = go.Figure()

# 1. Добавяне на оригиналните изолинии за Ei/Ed (H/D от 0 до 2)
if 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {level}',
            line=dict(width=2),
            xaxis='x1'
        ))

# 2. Добавяне на новите изолинии за sigma_r (нормализирани от 0 до 1)
if 'sigma_r' in df_new.columns:
    unique_sigma_r = sorted(df_new['sigma_r'].unique())
    for sigma in unique_sigma_r:
        df_level = df_new[df_new['sigma_r'] == sigma].sort_values(by='H/D')
        x_normalized = df_level['H/D'] / 2  # Нормализация 0-2 → 0-1
        fig.add_trace(go.Scatter(
            x=x_normalized,
            y=df_level['y'],
            mode='lines',
            name=f'σr/Ei = {sigma}',
            line=dict(width=2),  # Плътна линия
            xaxis='x2'
        ))

# Настройки на графиките и осите
fig.update_layout(
    width=750,  # Малко по-широко за да се съберат надписите
    height=700,
    # Първа ос X (горна) - H/D от 0 до 2
    xaxis=dict(
        title='H/D',
        dtick=0.2,
        range=[0, 2],
        domain=[0.1, 0.95],  # Оставете място за надписи
        anchor='y',
        position=0.95,
        showgrid=True
    ),
    # Втора ос X (долна) - нормализирана от 0 до 1
    xaxis2=dict(
        title='σr/Ei (H/D норм.)',
        dtick=0.1,
        range=[0, 1],
        overlaying='x',
        side='bottom',
        anchor='y',
        domain=[0.1, 0.95],  # Същата позиция като горната ос
        showgrid=False
    ),
    yaxis=dict(
        title='y',
        dtick=0.1,
        range=[0, 2.5],
        scaleanchor='x',
        scaleratio=1
    ),
    title='Изолинии с две хоризонтални оси',
    legend=dict(
        title='Легенда',
        yanchor="top",
        y=1.15,
        xanchor="right",
        x=1
    ),
    margin=dict(t=100, b=100)  # Добавен марж за надписите
)

st.plotly_chart(fig, use_container_width=True)
