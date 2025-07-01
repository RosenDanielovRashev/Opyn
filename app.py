import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Изолинии с фиксиран минимален x = 0 и мащаб 1:1")

df = pd.read_csv("danni.csv")
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
        range=[0, None],  # Минимална граница 0, максимална автоматично
        dtick=0.1,
        tickformat=".1f"
    ),
    yaxis=dict(
        title='y',
        dtick=0.1,
        tickformat=".1f",
        scaleanchor='x'
    ),
    title='Изолинии с начална точка x=0',
    legend=dict(title='Легенда')
)

st.plotly_chart(fig)
