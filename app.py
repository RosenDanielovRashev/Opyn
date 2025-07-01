import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Изолинии по подадени точки с контрол на мащаба по осите")

# Зареждане на CSV
df = pd.read_csv("danni.csv")

# Показване на примерни данни
st.write("Примерни данни:", df.head())

# Вземаме уникалните стойности на Ei/Ed
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
        dtick=0.1  # мащаб по x
    ),
    yaxis=dict(
        title='y',
        dtick=0.1  # мащаб по y
    ),
    title='Изолинии на Ei/Ed с фиксиран мащаб по осите',
    legend=dict(title='Легенда')
)

st.plotly_chart(fig)
