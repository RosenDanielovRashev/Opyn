import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr")

# --- Входни параметри за Esr ---
st.header("Изчисление на Esr")

# Вход: брой пластове
n = st.number_input("Брой пластове (n)", min_value=1, step=1, value=3)

h_values = []
E_values = []

st.markdown("### Въведи стойности за всеки пласт")

# Функция за преобразуване на цифра в долен индекс (пример: 3 → ₃)
def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

cols = st.columns(2)
for i in range(n):
    with cols[0]:
        h = st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        E = st.number_input(f"E{to_subscript(i+1)}", value=1000.0, step=0.1, key=f"E_{i}")
        E_values.append(E)

# Изчисляване на Esr и H
h_array = np.array(h_values)
E_array = np.array(E_values)

sum_h = h_array.sum()
weighted_sum = np.sum(E_array * h_array)
Esr = weighted_sum / sum_h if sum_h != 0 else 0

# Показване на формулата с LaTeX
st.latex(r"Esr = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}")

# Показване на формулата с конкретни стойности (пример за n пластове)
numerator = " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(n)])
denominator = " + ".join([f"{h_values[i]}" for i in range(n)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum}}}{{{sum_h}}} = {Esr:.3f}"
st.latex(formula_with_values)

st.latex(r"H = \sum_{i=1}^{n} h_i")
st.write(f"Обща дебелина H = {sum_h:.3f}")
st.write(f"Изчислено Esr = {Esr:.3f}")

# Въвеждане за алтернативно Ei с индекс n+1
alt_index = n + 1
Ei_alt = st.number_input(f"E (алтернативна стойност, E{to_subscript(alt_index)})", value=None, step=0.1)

if Ei_alt is not None and Ei_alt != 0:
    # Добавяме новия E към списъка с E_values
    E_values_alt = E_values + [Ei_alt]
    # Също добавяме един h с 0 (т.е. няма дебелина) - или може да добавиш дебелина по избор
    h_values_alt = h_values + [0]
    
    E_array_alt = np.array(E_values_alt)
    h_array_alt = np.array(h_values_alt)
    sum_h_alt = h_array_alt.sum()
    weighted_sum_alt = np.sum(E_array_alt * h_array_alt)
    
    # Ако добавеният h е 0, изчислението не се променя - може да сложиш h=4 по желание
    if sum_h_alt == 0:
        Esr_alt = 0
    else:
        Esr_alt = weighted_sum_alt / sum_h_alt
    
    numerator_alt = " + ".join([f"{E_values_alt[i]} \cdot {h_values_alt[i]}" for i in range(len(E_values_alt))])
    denominator_alt = " + ".join([f"{h_values_alt[i]}" for i in range(len(h_values_alt))])
    formula_with_values_alt = rf"Esr = \frac{{{numerator_alt}}}{{{denominator_alt}}} = \frac{{{weighted_sum_alt}}}{{{sum_h_alt}}} = {Esr_alt:.3f}"
    
    st.write(f"Обновено с алтернативен E{to_subscript(alt_index)}:")
    st.latex(formula_with_values_alt)
    st.write(f"Обновено Esr = {Esr_alt:.3f}")

# --- Номограма (графика) долу ---

# Зареждане на оригиналните данни
df_original = pd.read_csv("danni.csv")

# Зареждане на новите данни
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

fig = go.Figure()

# Добавяне на оригиналните изолинии за Ei/Ed
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

# Добавяне на новите изолинии за sr_Ei
if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'σsr/Ei = {sr_Ei}',
            line=dict(width=2)
        ))

# Празна следа за горната ос (xaxis2)
fig.add_trace(go.Scatter(
    x=[0,1],
    y=[None,None],
    mode='lines',
    xaxis='x2',
    showlegend=False,
    hoverinfo='skip'
))

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
        title='σₙ',
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
