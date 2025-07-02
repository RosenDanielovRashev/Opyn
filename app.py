import streamlit as st 
import pandas as pd
import plotly.graph_objects as go

st.title("Комбинирани изолинии")

# Зареждане на данни
try:
    df_original = pd.read_csv("danni.csv")
    df_new = pd.read_csv("Оразмеряване на опън за междинен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)
except FileNotFoundError as e:
    st.error(f"Грешка при зареждане на файл: {e}")
    st.stop()

# Създаване на фигура
fig = go.Figure()

# 1. Оригинални изолинии (Ei/Ed)
if 'Ei/Ed' in df_original.columns:
    for level in sorted(df_original['Ei/Ed'].unique()):
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values('H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {level}',
            line=dict(width=2)
        ))

# 2. Нови изолинии (σsr/Ei)
if 'sr_Ei' in df_new.columns:
    for sr_Ei in sorted(df_new['sr_Ei'].unique()):
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values('H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'σsr/Ei = {sr_Ei}',
            line=dict(width=2)
        ))

# Допълнителна ос x отгоре
fig.update_layout(
    xaxis2=dict(
        title="Нова ос",
        overlaying="x",
        side="top",
        range=[0, 1],  # Обхват 0-1
        matches="x",   # Синхронизиране с основната ос
        showgrid=False,
        tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Опционално: задаване на ticks
    ),
    xaxis=dict(
        title="H/D",
        range=[0, 2],
        dtick=0.2
    ),
    yaxis=dict(
        title="y",
        range=[0, 2.5],
        scaleanchor="x",
        scaleratio=1
    ),
    width=700,
    height=700,
    legend=dict(title="Легенда")
)

st.plotly_chart(fig, use_container_width=False)
