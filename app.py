import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Функция за долен индекс (subscript)
def to_subscript(number):
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscript_map)

st.title("Комбинирани изолинии")

# Входни параметри
st.header("Входни параметри")

num_layers = st.number_input("Брой пластове (n)", min_value=1, max_value=10, value=3, step=1)

h_values = []
E_values = []
for i in range(num_layers):
    h = st.number_input(f"h{i+1}", min_value=0.0, value=4.0, step=0.1, key=f"h{i+1}")
    E = st.number_input(f"E{i+1}", min_value=0.0, value=1000.0, step=10.0, key=f"E{i+1}")
    h_values.append(h)
    E_values.append(E)

# Алтернативен пласт n+1
h_extra = st.number_input(f"h{num_layers+1}", min_value=0.0, value=0.0, step=0.1, key="h_extra")
E_extra = st.number_input(
    f"E (алтернативна стойност, E{to_subscript(num_layers+1)})",
    min_value=0.0,
    value=1000.0,
    step=10.0,
    key="E_extra"
)

# Параметър D
D = st.number_input("D", min_value=0.1, value=1.0, step=0.1)

# Изчисления
Hn_minus_1 = sum(h_values[:-1]) if num_layers > 1 else 0
Hn = sum(h_values)
Hn_plus_extra = Hn + h_extra

# Esr (без последния пласт)
numerator = sum(E * h for E, h in zip(E_values, h_values))
Esr = numerator / Hn if Hn > 0 else 0
En = E_values[-1] if E_values else 0
Esr_over_En = Esr / En if En != 0 else 0
ratio = Hn / D if D != 0 else 0

# Формули
st.subheader("Изчисления и формули")

st.latex(r"H_{n-1} = " + " + ".join([f"h_{{{i+1}}}" for i in range(num_layers-1)]))
st.latex(f"{Hn_minus_1:.3f}")

st.latex(r"H_{n} = " + " + ".join([f"h_{{{i+1}}}" for i in range(num_layers)]))
st.latex(f"{Hn:.3f}")

st.latex(r"E_{sr} = \frac{" + " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(num_layers)]) +
         f"}}{{{Hn:.3f}}} = {Esr:.3f}")

st.latex(r"E_{" + f"{num_layers}" + r"} = " + f"{En:.3f}")

st.latex(r"\frac{E_{sr}}{E_{" + f"{num_layers}" + r"}} = " + f"{Esr:.3f} \div {En:.3f} = {Esr_over_En:.3f}")

st.latex(r"\frac{H_n}{D} = " + f"{Hn:.3f} \div {D:.3f} = {ratio:.3f}")

# Зареждане на CSV данни
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

# Създаване на графика
fig = go.Figure()

# Изолинии Ei/Ed
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

# Изолинии Esr/Ei (σ_r / E_i)
if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'σₙ / Eᵢ = {sr_Ei}',
            line=dict(width=2)
        ))

# Добавяне на вертикална линия от (Hn/D, 0) до (Hn/D, Esr/En)
fig.add_trace(go.Scatter(
    x=[ratio, ratio],
    y=[0, Esr_over_En],
    mode='lines',
    name="Линия Hn/D → Esr/En",
    line=dict(color='red', width=3, dash='dot'),
    opacity=0.6
))

# Настройки
fig.update_layout(
    width=800,
    height=700,
    xaxis=dict(
        title='H/D',
        dtick=0.1,
        range=[0, 2],
        constrain='domain'
    ),
    yaxis=dict(
        title='σₙ / Eᵢ',
        dtick=0.1,
        range=[0, 2.5],
        scaleanchor='x',
        scaleratio=1
    ),
    title='Комбинирани изолинии',
    legend=dict(title='Легенда')
)

st.plotly_chart(fig, use_container_width=False)
