import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междинен пласт от пътната конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# --- Инициализация на session_state ---
if "n" not in st.session_state:
    st.session_state.n = 3
if "h_values" not in st.session_state:
    st.session_state.h_values = [4.0] * st.session_state.n
if "E_values" not in st.session_state:
    st.session_state.E_values = [1000.0] * st.session_state.n
if "Ed" not in st.session_state:
    st.session_state.Ed = 100.0
if "D" not in st.session_state:
    st.session_state.D = 32.04

# --- Управление на броя пластове с бутони ---
col_buttons = st.columns([1,1,6])

with col_buttons[0]:
    if st.button("Назад"):
        if st.session_state.n > 2:
            st.session_state.n -= 1
            st.session_state.h_values = st.session_state.h_values[:st.session_state.n]
            st.session_state.E_values = st.session_state.E_values[:st.session_state.n]

with col_buttons[1]:
    if st.button("Напред"):
        st.session_state.n += 1
        st.session_state.h_values.append(4.0)
        st.session_state.E_values.append(1000.0)

# --- Входни параметри ---
st.write(f"**Брой пластове (n): {st.session_state.n}**")

st.session_state.D = st.selectbox("Избери D", options=[32.04, 34.0], index=[32.04, 34.0].index(st.session_state.D))

# --- Въвеждане на h_i и E_i за всеки пласт ---
st.markdown("### Въведи стойности за всеки пласт")
h_new = []
E_new = []
cols = st.columns(2)
for i in range(st.session_state.n):
    with cols[0]:
        h_val = st.number_input(f"h{to_subscript(i+1)}", value=st.session_state.h_values[i], step=0.1, key=f"h_{i}")
    with cols[1]:
        E_val = st.number_input(f"E{to_subscript(i+1)}", value=st.session_state.E_values[i], step=0.1, key=f"E_{i}")
    h_new.append(h_val)
    E_new.append(E_val)

st.session_state.h_values = h_new
st.session_state.E_values = E_new

# Ed може да се променя по всяко време
st.session_state.Ed = st.number_input("Ed", value=st.session_state.Ed, step=0.1)

# --- Бутон за изчисления и визуализация ---
if st.button("Покажи графиката"):
    n = st.session_state.n
    D = st.session_state.D
    h_array = np.array(st.session_state.h_values)
    E_array = np.array(st.session_state.E_values)
    Ed = st.session_state.Ed

    # Изчисляване на Esr за първите n-1 пласта
    sum_h_n_1 = h_array[:-1].sum()
    weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
    Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0

    H_n = h_array.sum()
    H_n_1 = sum_h_n_1

    H_n_1_r = round(H_n_1, 3)
    H_n_r = round(H_n, 3)
    Esr_r = round(Esr, 3)
    ratio = H_n / D if D != 0 else 0
    ratio_r = round(ratio, 3)

    st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
    h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n-1)])
    st.latex(r"H_{n-1} = " + h_terms)
    st.write(f"H{to_subscript(n-1)} = {H_n_1_r}")

    st.latex(r"H_n = \sum_{i=1}^n h_i")
    h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n)])
    st.latex(r"H_n = " + h_terms_n)
    st.write(f"H{to_subscript(n)} = {H_n_r}")

    st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")

    numerator = " + ".join([f"{round(st.session_state.E_values[i],3)} \cdot {round(st.session_state.h_values[i],3)}" for i in range(n-1)])
    denominator = " + ".join([f"{round(st.session_state.h_values[i],3)}" for i in range(n-1)])
    formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{round(weighted_sum_n_1,3)}}}{{{round(sum_h_n_1,3)}}} = {Esr_r}"
    st.latex(formula_with_values)

    ratio_display = rf"\frac{{H_n}}{{D}} = \frac{{{H_n_r}}}{{{round(D,3)}}} = {ratio_r}"
    st.latex(ratio_display)

    En = st.session_state.E_values[-1]
    En_r = round(En, 3)

    st.markdown("### Изчисления с последен пласт")

    st.latex(r"E_{" + str(n) + r"} = " + f"{En_r}")

    Esr_over_En = Esr / En if En != 0 else 0
    Esr_over_En_r = round(Esr_over_En, 3)
    st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr_r}" + "}{" + f"{En_r}" + "} = " + f"{Esr_over_En_r}")

    En_over_Ed = En / Ed if Ed != 0 else 0
    En_over_Ed_r = round(En_over_Ed, 3)
    st.latex(r"\frac{E_{" + str(n) + r"}}{E_d} = \frac{" + f"{En_r}" + "}{" + f"{Ed_r}" + "} = " + f"{En_over_Ed_r}")

    # Тук можеш да добавиш останалата част от твоя код за четене на CSV файлове и графика
    st.info("Тук постави кода си за графиката...")

    # Например:
    # df_original = pd.read_csv("danni.csv")
    # ... (твоя оригинален код за графика)

    # st.plotly_chart(fig)

