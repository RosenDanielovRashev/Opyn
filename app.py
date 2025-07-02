import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr и H_n/D")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[34.0, 32.04], index=0)

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

sum_h_n_1 = h_array[:-1].sum()
weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0

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
numerator = " + ".join([f"{E_values[i]} · {h_values[i]}" for i in range(n-1)])
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
        fig.add_trace(go.Scatter(x=df_level['H/D'], y=df_level['y'], mode='lines', name=f'Ei/Ed = {level}', line=dict(width=2)))

if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        fig.add_trace(go.Scatter(x=df_level['H/D'], y=df_level['y'], mode='lines', name=f'Esr/Ei = {sr_Ei}', line=dict(width=2)))

# Интерполация и чертане на точки и линии
# (същата логика като досега, но накрая добавяме вертикалната линия нагоре и хоризонталното разстояние)

# >>> След интерполация на точките и намиране на x_interp_EiEd и interp_point[1]:

# Добавяне на точка на пресичане с Ei/Ed
fig.add_trace(go.Scatter(x=[x_interp_EiEd], y=[interp_point[1]], mode='markers', marker=dict(color='orange', size=10), name='Пресечна точка с Ei/Ed'))

# Вертикална линия нагоре
y_top = max(df_original['y'].max(), df_new['y'].max())
fig.add_trace(go.Scatter(x=[x_interp_EiEd, x_interp_EiEd], y=[interp_point[1], y_top], mode='lines', line=dict(color='orange', dash='dot'), name='Вертикална линия нагоре'))

# Анотация за хоризонтално разстояние по втората ос (xaxis2)
horizontal_distance = round(x_interp_EiEd, 3)
fig.add_annotation(
    x=x_interp_EiEd,
    y=y_top + 0.1,
    text=f"Δx = {horizontal_distance}",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=1,
    arrowcolor="orange",
    ax=0,
    ay=y_top + 0.1,
    font=dict(color="orange"),
    xref='x',
    yref='y',
    axref='x',
    ayref='y'
)

# Ос x2
fig.update_layout(
    xaxis_title='H/D',
    yaxis_title='y',
    title='Изолинии с интерполации',
    showlegend=False,
    xaxis2=dict(
        overlaying='x',
        side='top',
        range=[0, 1],
        showgrid=False,
        zeroline=False,
        tickmode='auto',
        ticks='outside'
    )
)

st.plotly_chart(fig, use_container_width=True)
