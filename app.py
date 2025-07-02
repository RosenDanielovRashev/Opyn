import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Комбинирани изолинии с две X оси")

# Зареждане на данни
try:
    df_original = pd.read_csv("danni.csv")
except Exception:
    st.error("Грешка при зареждане на 'danni.csv'")
    st.stop()

try:
    df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)
except Exception:
    st.error("Грешка при зареждане на новия CSV")
    st.stop()

fig = go.Figure()

# Функция за добавяне на линии към фигурата
def add_isoline(df, column_level, label_prefix, line_style='solid', xaxis_name='x'):
    levels = sorted(df[column_level].dropna().unique())
    for level in levels:
        df_level = df[df[column_level] == level].dropna(subset=['H/D', 'y']).sort_values('H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f"{label_prefix} = {level}",
            line=dict(dash=line_style),
            xaxis=xaxis_name
        ))

# Добавяне на оригиналните изолинии Ei/Ed (долна ос)
if 'Ei/Ed' in df_original.columns:
    add_isoline(df_original, 'Ei/Ed', 'Ei/Ed', line_style='solid', xaxis_name='x')

# Добавяне на новите изолинии sr_Ei (долна ос, с различен стил)
if 'sr_Ei' in df_new.columns:
    add_isoline(df_new, 'sr_Ei', 'σsr/Ei', line_style='dot', xaxis_name='x')

# Добавяне на изолинии за σr = (H/D) / 2 (горна ос)
if 'sr_Ei' in df_new.columns:
    levels = sorted(df_new['sr_Ei'].dropna().unique())
    for level in levels:
        df_level = df_new[df_new['sr_Ei'] == level].dropna(subset=['H/D', 'y']).sort_values('H/D')
        sigma_r = df_level['H/D'] / 2  # Трансформация за горната ос
        fig.add_trace(go.Scatter(
            x=sigma_r,
            y=df_level['y'],
            mode='lines',
            name=f"σr = {level} (горна ос)",
            line=dict(dash='dash'),
            xaxis='x2'
        ))

# Настройка на layout с две X оси
fig.update_layout(
    width=800,
    height=700,
    title="Комбинирани изолинии с H/D (долу) и σr (горе)",

    xaxis=dict(
        title="H/D",
        range=[0, 2],
        domain=[0, 1],
        dtick=0.2,
        showgrid=True,
        zeroline=False
    ),

    xaxis2=dict(
        title="σr",
        range=[0, 1],
        side='top',
        overlaying='x',
        dtick=0.1,
        showgrid=False,
        zeroline=False,
        ticks='outside',
        showline=True
    ),

    yaxis=dict(
        title="y",
        range=[0, 2.5],
        scaleanchor='x',
        scaleratio=1,
        dtick=0.1,
        zeroline=False
    ),

    legend=dict(
        title=dict(text='Легенда'),
        font=dict(size=10)
    )
)

st.plotly_chart(fig, use_container_width=False)
