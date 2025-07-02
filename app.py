import streamlit as st 
import pandas as pd
import plotly.graph_objects as go

st.title("Изолинии с две хоризонтални оси")

# Зареждане на данни
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'?sr/Ei': 'sigma_r'}, inplace=True)

# Създаване на фигура
fig = go.Figure()

# 1. Оригинални изолинии (Ei/Ed)
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

# 2. Нови изолинии (σr/Ei)
if 'sigma_r' in df_new.columns:
    unique_sigma_r = sorted(df_new['sigma_r'].unique())
    for sigma in unique_sigma_r:
        df_level = df_new[df_new['sigma_r'] == sigma].sort_values(by='H/D')
        x_normalized = df_level['H/D'] / 2
        fig.add_trace(go.Scatter(
            x=x_normalized,
            y=df_level['y'],
            mode='lines',
            name=f'σr/Ei = {sigma}',
            line=dict(width=2),
            xaxis='x2'
        ))

# Настройки на графиките
fig.update_layout(
    width=700,
    height=700,
    # Първа ос X (горна)
    xaxis=dict(
        title='H/D',
        dtick=0.2,
        range=[0, 2],
        anchor='y',
        position=1.0,  # Точна позиция
        showgrid=True
    ),
    # Втора ос X (долна)
    xaxis2=dict(
        title='σr/Ei',
        dtick=0.1,
        range=[0, 1],
        overlaying='x',
        side='bottom',
        anchor='free',
        position=0.0,  # Точна позиция под графиката
        showgrid=False
    ),
    yaxis=dict(
        title='y',
        dtick=0.1,
        range=[0, 2.5],
        scaleanchor='x',
        scaleratio=1,
        domain=[0.0, 0.95]  # Оставя място за долната ос
    ),
    title='Изолинии с две хоризонтални оси',
    legend=dict(
        title='Легенда',
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(b=60)  # Добавя място за долната ос
)

st.plotly_chart(fig, use_container_width=False)
