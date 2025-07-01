import streamlit as st 
import pandas as pd
import plotly.graph_objects as go

st.title("Изолинии с реално съотношение 1:1")

# Зареждане на данни
df = pd.read_csv("danni.csv")

# Показване на данни
st.write("Примерни данни:", df.head())

unique_levels = sorted(df['Ei/Ed'].unique())

fig = go.Figure()

for level in unique_levels:
    df_level = df[df['Ei/Ed'] == level].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df_level['H/D'],
        y=df_level['y'],
        mode='lines',
        name=f'Ei/Ed = {level}',
        line=dict(width=2)
    ))

fig.update_layout(
    width=700,   # Ширина на графиката в пиксели
    height=700,  # Височина на графиката в пиксели (за 1:1 квадрат)
    xaxis=dict(
        title='H/D',
        dtick=0.1,
        range=[0, 2],
        constrain='domain',  # опционално, за да се ограничи разтягането
    ),
    yaxis=dict(
        title='y',
        dtick=0.1,
        range=[0, 2.7],
        scaleanchor='x',  # Скалата по y е привързана към x -> реален мащаб 1:1
        scaleratio=1      # гарантира съотношение 1:1 (понякога излишно, но няма да навреди)
    ),
    title='Изолинии с реален мащаб 1:1',
    legend=dict(title='Легенда')
)

st.plotly_chart(fig, use_container_width=False)  # с False, за да не се мащабира според контейнера
