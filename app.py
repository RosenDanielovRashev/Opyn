import streamlit as st 
import pandas as pd
import plotly.graph_objects as go

st.title("Комбинирани изолинии")

# Зареждане на оригиналните данни
df_original = pd.read_csv("danni.csv")

# Зареждане на новите данни
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

# Създаване на фигура
fig = go.Figure()

# 1. Добавяне на оригиналните изолинии за Ei/Ed
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

# 2. Добавяне на новите изолинии за sr_Ei
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

# Добавяне на празна следа за визуализиране на горната ос (xaxis2)
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[None, None],
    mode='lines',
    xaxis='x2',
    showlegend=False,
    hoverinfo='skip'
))

# Настройки на графиката с добавена горна ос с LaTeX заглавие
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
        title=r'$\sigma_r$',  # <-- LaTeX заглавие на новата ос
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

# Показване на графиката
st.plotly_chart(fig, use_container_width=False)
