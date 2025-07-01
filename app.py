import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Изолинии по зададени точки (без интерполация)")

# Зареждане на CSV
df = pd.read_csv("danni.csv")

# Показваме част от данните за преглед
st.write("Примерни данни:", df.head())

# Извличаме уникалните стойности на Ei/Ed
unique_levels = sorted(df['Ei/Ed'].unique())

# Създаваме графиката
fig = go.Figure()

# За всяка изолиния (всяка стойност на Ei/Ed)
for level in unique_levels:
    df_level = df[df['Ei/Ed'] == level].sort_values(by='H/D')  # или by='y' според структурата

    fig.add_trace(go.Scatter(
        x=df_level['H/D'],
        y=df_level['y'],
        mode='lines',
        name=f'Ei/Ed = {level}',
        line=dict(width=2)
    ))

# Оформление
fig.update_layout(
    xaxis_title='H/D',
    yaxis_title='y',
    title='Изолинии на Ei/Ed (по подадени точки)',
    legend=dict(title='Легенда'),
    xaxis=dict(dtick=0.1)
)

st.plotly_chart(fig)
