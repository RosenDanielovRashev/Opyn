import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr и H_n/D")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Входни параметри
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[34.0, 32.04], index=0)

# Въвеждане на h_i и E_i за всеки пласт
st.markdown("### Въведи стойности за всеки пласт")
h_values = []
E_values = []
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

# Изчисляване на Esr за първите n-1 пласта
sum_h_n_1 = h_array[:-1].sum()
weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0

# Изчисляване на H_n и H_{n-1}
H_n = h_array.sum()
H_n_1 = sum_h_n_1

st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n-1)])
st.latex(r"H_{n-1} = " + h_terms)
st.write(f"H{to_subscript(n-1)} = {H_n_1:.3f}")

st.latex(r"H_n = \sum_{i=1}^n h_i")
h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n)])
st.latex(r"H_n = " + h_terms_n)
st.write(f"H{to_subscript(n)} = {H_n:.3f}")

st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")

numerator = " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(n-1)])
denominator = " + ".join([f"{h_values[i]}" for i in range(n-1)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum_n_1}}}{{{sum_h_n_1}}} = {Esr:.3f}"
st.latex(formula_with_values)

ratio = H_n / D if D != 0 else 0
st.latex(r"\frac{H_n}{D} = \frac{" + f"{H_n:.3f}" + "}{" + f"{D:.3f}" + "} = " + f"{ratio:.3f}" )

Ed = st.number_input("Ed", value=1000.0, step=0.1)

En = E_values[-1]

st.markdown("### Изчисления с последен пласт")

st.latex(r"E_{" + str(n) + r"} = " + f"{En:.3f}")

Esr_over_En = Esr / En if En != 0 else 0
st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr:.3f}" + "}{" + f"{En:.3f}" + "} = " + f"{Esr_over_En:.3f}")

En_over_Ed = En / Ed if Ed != 0 else 0
st.latex(r"\frac{E_{" + str(n) + r"}}{E_d} = \frac{" + f"{En:.3f}" + "}{" + f"{Ed:.3f}" + "} = " + f"{En_over_Ed:.3f}")

# Зареждане на данни и построяване на графика
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

# ... (тук остава целият ти съществуващ код за интерполации, без промяна) ...

# --- Добавяне на прозрачна линия, за да се визуализира горната ос (от 0 до 1)
fig.add_trace(go.Scatter(
    x=np.linspace(0, 1, 100),
    y=[fig.layout.yaxis.range[0] if fig.layout.yaxis.range else 0]*100,  # долна част по y
    mode='lines',
    line=dict(color='rgba(0,0,0,0.1)', width=3),
    xaxis='x2',
    showlegend=False,
    hoverinfo='skip'
))

fig.update_layout(
    xaxis_title='H/D',
    yaxis_title='y',
    title='Изолинии с интерполации',
    showlegend=False,
    xaxis2=dict(
        overlaying='x',
        side='top',
        range=[0, 2],                 # основен диапазон на X (примерно)
        tickvals=[0, 0.5, 1, 1.5, 2],# позиции на отметките по основната ос
        ticktext=['0', '0.25', '0.5', '0.75', '1'],  # текст по новата ос (нормализирани)
        ticks='outside',
        tickfont=dict(size=10),       # по-малък шрифт, за да не се препокриват
        tickangle=45,                 # завъртени на 45 градуса
        position=1.05                # изместване нагоре, за да има разстояние от основната ос
    )
)

st.plotly_chart(fig, use_container_width=True)
