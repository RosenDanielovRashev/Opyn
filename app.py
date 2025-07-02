import streamlit as st 
import pandas as pd
import plotly.graph_objects as go

st.title("Комбинирани изолинии с активна горна ос σr")

# Зареждане на оригиналните данни
try:
    df_original = pd.read_csv("danni.csv")
except FileNotFoundError:
    st.error("Файлът danni.csv не е намерен.")
    st.stop()

# Зареждане на новите данни
try:
    df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)
except FileNotFoundError:
    st.error("Файлът с новите данни не е намерен.")
    st.stop()

# Създаване на фигура
fig = go.Figure()

# 1. Оригинални изолинии спрямо H/D
if 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        df_level = df_level.dropna(subset=['H/D', 'y'])
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {level}',
            line=dict(width=2),
            xaxis='x'  # основната ос
        ))

# 2. Изолинии от новите данни (σsr/Ei), спрямо H/D
if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        df_level = df_level.dropna(subset=['H/D', 'y'])
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'σsr/Ei = {sr_Ei}',
            line=dict(width=2, dash='dot'),
            xaxis='x'  # основна ос
        ))

# 3. Криви спрямо σr = H/D / 2 → изобразени върху xaxis2
if 'sr_Ei' in df_new.columns:
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        df_level = df_level.dropna(subset=['H/D', 'y'])
        sigma_r = df_level['H/D'] / 2  # нова стойност по σr

        fig.add_trace(go.Scatter(
            x=sigma_r,
            y=df_level['y'],
            mode='lines',
            name=f'σr = {sr_Ei} (горна ос)',
            line=dict(width=1, dash='dash'),
            xaxis='x2'  # вързано към горната ос
        ))

# Настройки на графиката
fig.update_layout(
    width=800,
    height=700,

    # Основна долна ос H/D
    xaxis=dict(
        title='H/D',
        range=[0, 2],
        dtick=0.2,
        domain=[0, 1]
    ),

    # Горна ос σr (xaxis2) със стойности от 0 до 1
    xaxis2=dict(
        title='σr',
        range=[0, 1],
        dtick=0.1,
        side='top',
        overlaying='x',
        showline=True,
        ticks='outside',
        showgrid=False,
        titlefont=dict(size=14),
        tickfont=dict(size=12)
    ),

    # Вертикална ос
    yaxis=dict(
        title='y',
        range=[0, 2.5],
        dtick=0.1,
        scaleanchor='x',
        scaleratio=1
    ),

    title='Комбинирани изолинии с H/D и σr',
    legend=dict(title='Легенда', font=dict(size=10))
)

# Показване на графиката
st.plotly_chart(fig, use_container_width=False)
