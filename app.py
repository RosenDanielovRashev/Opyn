import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Изолинии с фиксиран диапазон и реален мащаб 1:1")

# Зареждане на данни
df = pd.read_csv("danni.csv")

# Показваме примерни данни
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
    xaxis=dict(
        title='H/D',
        range=[0, 2],
        dtick=0.1
    ),
    yaxis=dict(
        title='y',
        range=[0, 2.7],
        dtick=0.1,
        scaleanchor='x'
    ),
    title='Изолинии с фиксиран диапазон и мащаб 1:1',
    legend=dict(title='Легенда')
)

st.plotly_chart(fig)
