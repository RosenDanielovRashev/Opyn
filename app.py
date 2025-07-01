import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Изолинии за всички стойности на Ei/Ed (без интерполация)")

# Зареждане на данни
df = pd.read_csv("danni.csv")

# Проверяваме колоните
st.write("Колони:", df.columns.tolist())

# Получаваме всички уникални стойности на Ei/Ed
unique_levels = sorted(df['Ei/Ed'].unique())

# Създаваме фигура
fig = go.Figure()

# За всяка стойност на Ei/Ed:
for level in unique_levels:
    df_level = df[df['Ei/Ed'] == level]

    # Създаваме pivot таблица: редове = y, колони = H/D
    pivot = df_level.pivot_table(index='y', columns='H/D', values='Ei/Ed')

    # Проверка дали имаме пълна мрежа (иначе прескачаме)
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        continue

    x = pivot.columns.values       # H/D
    y = pivot.index.values         # y
    z = pivot.values               # матрица със стойности (всички равни на level)

    # Добавяме контур само за конкретното ниво
    fig.add_trace(go.Contour(
        z=z,
        x=x,
        y=y,
        showscale=False,  # без цветна скала
        contours=dict(
            coloring='lines',
            showlabels=True,
            start=level,
            end=level,
            size=1
        ),
        line=dict(width=2),
        name=f'Ei/Ed = {level}'
    ))

# Оформление
fig.update_layout(
    xaxis_title='H/D',
    yaxis_title='y',
    title='Изолинии на Ei/Ed (само реални точки)',
    legend=dict(x=0.7, y=0.95),
    xaxis=dict(dtick=0.1)
)

st.plotly_chart(fig)
