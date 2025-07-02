import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# --- Входни параметри за Esr ---
st.header("Изчисление на Esr")

n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)

h_values = []
E_values = []

st.markdown("### Въведи стойности за всеки пласт")

cols = st.columns(2)
for i in range(n):
    with cols[0]:
        h = st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        E = st.number_input(f"E{to_subscript(i+1)}", value=1000.0, step=0.1, key=f"E_{i}")
        E_values.append(E)

h_array = np.array(h_values)
E_array = np.array(E_values)

sum_h = h_array[:-1].sum()
weighted_sum = np.sum(E_array[:-1] * h_array[:-1])
Esr = weighted_sum / sum_h if sum_h != 0 else 0

st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")

numerator = " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(n-1)])
denominator = " + ".join([f"{h_values[i]}" for i in range(n-1)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum}}}{{{sum_h}}} = {Esr:.3f}"
st.latex(formula_with_values)

st.latex(r"H = \sum_{i=1}^{n-1} h_i")
st.write(f"Обща дебелина H = {sum_h:.3f}")
st.write(f"Изчислено Esr = {Esr:.3f}")

h_next = h_array[-1]
st.write(f"h{to_subscript(n)} = {h_next:.3f}")

Ed = st.number_input("Ed", value=100.0, step=0.1)

D = st.selectbox("Избери D", options=[34.0, 32.04], index=0)

# Зареждане на данни за диаграмата
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

fig = go.Figure()

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

if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Esr/Ei = {sr_Ei}',
            line=dict(width=2)
        ))

fig.update_layout(
    width=700,
    height=700,
    margin=dict(t=120),  # Оставяме място за горната ос
    xaxis=dict(
        title='H/D',
        dtick=0.1,
        range=[0, 2],
        domain=[0, 1],
        anchor='y',
        tickangle=0  # Без въртене на числата
    ),
    xaxis2=dict(
        title=r'$\sigma_n$',
        overlaying='x',
        side='top',
        range=[0, 1],
        tickmode='linear',
        tick0=0,
        dtick=0.1,
        showgrid=False,
        anchor='y',
        position=1.0,
        tickangle=0  # Без въртене на числата
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

st.plotly_chart(fig, use_container_width=False)
