import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr")

# Функция за индекси с долен индекс
def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# --- Входни параметри за Esr ---
st.header("Изчисление на Esr")

# Вход: брой пластове
n = st.number_input("Брой пластове (n)", min_value=1, step=1, value=3)

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

sum_h = h_array.sum()
weighted_sum = np.sum(E_array * h_array)
Esr = weighted_sum / sum_h if sum_h != 0 else 0

# Формула Esr - LaTeX
st.latex(r"Esr = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}")

numerator = " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(n)])
denominator = " + ".join([f"{h_values[i]}" for i in range(n)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum}}}{{{sum_h}}} = {Esr:.3f}"
st.latex(formula_with_values)

st.latex(r"H = \sum_{i=1}^{n} h_i")
st.write(f"Обща дебелина H = {sum_h:.3f}")
st.write(f"Изчислено Esr = {Esr:.3f}")

# Вход за h_{n+1}
h_next = st.number_input(
    f"h{to_subscript(n+1)}",
    value=0.0,
    step=0.1
)

H_next = sum_h + h_next
st.write(f"H{to_subscript(n+1)} = H + h{to_subscript(n+1)} = {sum_h:.3f} + {h_next:.3f} = {H_next:.3f}")

# Вход за Ed (числов)
Ed = st.number_input(
    "Ed",
    value=1000.0,
    step=0.1
)

# Вход за E (алтернативна стойност, индекс n+1)
Ei_alt = st.number_input(
    f"E (алтернативна стойност, E{to_subscript(n+1)})",
    value=1000.0,
    step=0.1
)

# Зареждане на данни за диаграмата
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

# Създаване на фигура
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
            name=f'σₙ/Ei = {sr_Ei}',
            line=dict(width=2)
        ))

# Добавяне на втората x-ос (σₙ)
fig.update_layout(
    width=700,
    height=700,
    xaxis=dict(
        title='H/D',
        dtick=0.1,
        range=[0, 2],
        constrain='domain',
        domain=[0, 1]
    ),
    xaxis2=dict(
        title=r'$\sigma_n$',
        overlaying='x',
        side='top',
        range=[0, 1],
        tickmode='linear',
        tick0=0,
        dtick=0.1,
        showgrid=False
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
