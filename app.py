import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr")

# Вход: брой пластове
n = st.number_input("Брой пластове (n)", min_value=1, step=1, value=3)

h_values = []
E_values = []

st.markdown("### Въведи стойности за всеки пласт")

cols = st.columns(2)
for i in range(n):
    with cols[0]:
        h = st.number_input(f"h_{i+1}", value=1.0, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        E = st.number_input(f"E_{i+1}", value=1.0, step=0.1, key=f"E_{i}")
        E_values.append(E)

# Отделно въвеждане за Ei на последния пласт (ако искаш да го зададеш различно)
Ei_last = st.number_input(f"E (последен пласт, E_{n+1})", value=None, step=0.1)

# Ако е зададено Ei_last, замени последния Ei с него
if Ei_last is not None and Ei_last != 0:
    if len(E_values) == n:
        E_values[-1] = Ei_last

# Изчисляване на Esr и H
h_array = np.array(h_values)
E_array = np.array(E_values)

sum_h = h_array.sum()
Esr = np.sum(E_array * h_array) / sum_h if sum_h != 0 else 0

# Показване на формулите с LaTeX
st.latex(r"Esr = \frac{\sum_{i=1}^{n} (E_i \cdot h_i)}{\sum_{i=1}^{n} h_i}")
st.latex(r"H = \sum_{i=1}^{n} h_i")

# Показване на резултатите
st.write(f"Обща дебелина H = {sum_h:.3f}")
st.write(f"Изчислено Esr = {Esr:.3f}")

# (тук може да сложиш останалата част от твоя код за визуализация и т.н.)
