import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междиен пласт от пътната конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Въвеждаш броя пластове
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)

# Инициализация на session_state при първо зареждане или промяна на n
if "h_values" not in st.session_state or len(st.session_state.h_values) != n:
    st.session_state.h_values = [4.0] * n
if "E_values" not in st.session_state or len(st.session_state.E_values) != n:
    st.session_state.E_values = [1000.0] * n
if "Ed_values" not in st.session_state or len(st.session_state.Ed_values) != n:
    st.session_state.Ed_values = [100.0] * n
if "current_layer" not in st.session_state:
    st.session_state.current_layer = 0

# Избор на текущ пласт с бутоните "Назад" и "Напред"
col1, col2, col3 = st.columns([1,1,6])
with col1:
    if st.button("Назад") and st.session_state.current_layer > 0:
        st.session_state.current_layer -= 1
with col2:
    if st.button("Напред") and st.session_state.current_layer < n-1:
        st.session_state.current_layer += 1

st.markdown(f"### Текущ пласт: {st.session_state.current_layer + 1} от {n}")

# Редактиране на h и E за всички пластове (непроменяеми, само визуализация)
st.markdown("#### Въведени стойности за всички пластове (не могат да се променят):")
cols_h = st.columns(n)
cols_E = st.columns(n)
for i in range(n):
    with cols_h[i]:
        st.write(f"h{to_subscript(i+1)} = {st.session_state.h_values[i]}")
    with cols_E[i]:
        st.write(f"E{to_subscript(i+1)} = {st.session_state.E_values[i]}")

# Позволяваме да се редактира само Ed за текущия пласт
st.markdown(f"### Редактиране на Ed за пласт {st.session_state.current_layer + 1}")
new_Ed = st.number_input(
    f"Ed за пласт {to_subscript(st.session_state.current_layer + 1)}", 
    value=st.session_state.Ed_values[st.session_state.current_layer], 
    step=0.1
)
st.session_state.Ed_values[st.session_state.current_layer] = new_Ed

# Параметър D (по избор)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

# Бутон за изчисления и визуализация (взима всички стойности)
if st.button("Покажи графиката"):
    h_array = np.array(st.session_state.h_values)
    E_array = np.array(st.session_state.E_values)
    Ed_array = np.array(st.session_state.Ed_values)

    # Примерна логика за изчисления, използвайки средната Ed
    Ed_mean = np.mean(Ed_array)

    # Изчисляване на Esr за първите n-1 пласта
    sum_h_n_1 = h_array[:-1].sum()
    weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
    Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0

    # Изчисляване на H_n и H_{n-1}
    H_n = h_array.sum()
    H_n_1 = sum_h_n_1

    # Закръгляне
    H_n_1_r = round(H_n_1, 3)
    H_n_r = round(H_n, 3)
    Esr_r = round(Esr, 3)
    ratio = H_n / D if D != 0 else 0
    ratio_r = round(ratio, 3)
    Ed_r = round(Ed_mean, 3)

    st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
    h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n-1)])
    st.latex(r"H_{n-1} = " + h_terms)
    st.write(f"H{to_subscript(n-1)} = {H_n_1_r}")

    st.latex(r"H_n = \sum_{i=1}^n h_i")
    h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n)])
    st.latex(r"H_n = " + h_terms_n)
    st.write(f"H{to_subscript(n)} = {H_n_r}")

    st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")

    numerator = " + ".join([f"{round(E_array[i],3)} \cdot {round(h_array[i],3)}" for i in range(n-1)])
    denominator = " + ".join([f"{round(h_array[i],3)}" for i in range(n-1)])
    formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{round(weighted_sum_n_1,3)}}}{{{round(sum_h_n_1,3)}}} = {Esr_r}"
    st.latex(formula_with_values)

    ratio_display = rf"\frac{{H_n}}{{D}} = \frac{{{H_n_r}}}{{{round(D,3)}}} = {ratio_r}"
    st.latex(ratio_display)

    En = E_array[-1]
    En_r = round(En, 3)

    st.markdown("### Изчисления с последен пласт")

    st.latex(r"E_{" + str(n) + r"} = " + f"{En_r}")

    Esr_over_En = Esr / En if En != 0 else 0
    Esr_over_En_r = round(Esr_over_En, 3)
    st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr_r}" + "}{" + f"{En_r}" + "} = " + f"{Esr_over_En_r}")

    En_over_Ed = En / Ed_mean if Ed_mean != 0 else 0
    En_over_Ed_r = round(En_over_Ed, 3)
    st.latex(r"\frac{E_{" + str(n) + r"}}{E_d} = \frac{" + f"{En_r}" + "}{" + f"{Ed_r}" + "} = " + f"{En_over_Ed_r}")

    # --- Тук сложи кода за графиката както е в твоя пример ---

    st.info("Графиката е тук (постави твоя код за графиката)")

    # Можеш да добавиш същия код за графика и CSV четене както имаш

