import streamlit as st 
import pandas as pd
import plotly.graph_objects as go

st.title("Изолинии с две хоризонтални оси")

# Зареждане на данни
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

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
            xaxis='x1'  # Свързано с първата ос X
        ))

# 2. Добавяне на новите изолинии за sr_Ei (нормализирани от 0 до 1)
if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        # Нормализираме H/D стойностите от 0-2 към 0-1
        x_normalized = df_level['H/D'] / 2  
        fig.add_trace(go.Scatter(
            x=x_normalized,
            y=df_level['y'],
            mode='lines',
            name=f'σsr/Ei = {sr_Ei}',
            line=dict(width=2, dash='dot'),
            xaxis='x2'  # Свързано с втората ос X
        ))

# Настройки на графиките и осите
fig.update_layout(
    width=700,
    height=700,
    # Първа ос X (горна) - H/D от 0 до 2
    xaxis=dict(
        title='H/D',
        dtick=0.2,
        range=[0, 2],
        constrain='domain',
        anchor='y',
        position=0.95,  # Позициониране на горната ос
        showgrid=True
    ),
    # Втора ос X (долна) - нормализирана от 0 до 1
    xaxis2=dict(
        title='Нормализирано H/D',
        dtick=0.1,
        range=[0, 1],
        overlaying='x',
        side='bottom',  # Показва се отдолу
        anchor='y',
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
    legend=dict(title='Легенда')
)

st.plotly_chart(fig, use_container_width=False)
