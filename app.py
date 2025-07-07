import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междиен пласт от пътнатата конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Входни параметри
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

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

Ed = st.number_input("Ed", value=100.0, step=0.1)
Ed_r = round(Ed, 3)

# Инициализация на текущия пласт в сесията
if 'current_layer' not in st.session_state:
    st.session_state.current_layer = n

# Навигационни бутони
col1, col2 = st.columns(2)
with col1:
    if st.button("Назад"):
        if st.session_state.current_layer > 1:
            st.session_state.current_layer -= 1
with col2:
    if st.button("Напред"):
        if st.session_state.current_layer < n:
            st.session_state.current_layer += 1

current_layer = st.session_state.current_layer
st.markdown(f"### Изчисления за пласт {current_layer}")

h_array = np.array(h_values)
E_array = np.array(E_values)

# Изчисляване на Esr за първите current_layer - 1 пласта
if current_layer > 1:
    sum_h_n_1 = h_array[:current_layer-1].sum()
    weighted_sum_n_1 = np.sum(E_array[:current_layer-1] * h_array[:current_layer-1])
    Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0
else:
    sum_h_n_1 = 0
    weighted_sum_n_1 = 0
    Esr = 0

# Изчисляване на H_n и H_{n-1}
H_n = h_array[:current_layer].sum()
H_n_1 = sum_h_n_1

# Закръгляне
H_n_1_r = round(H_n_1, 3)
H_n_r = round(H_n, 3)
Esr_r = round(Esr, 3)
ratio = H_n / D if D != 0 else 0
ratio_r = round(ratio, 3)

st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
if current_layer > 1:
    h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(current_layer-1)])
    st.latex(r"H_{n-1} = " + h_terms)
else:
    st.write("Няма пластове преди текущия.")
st.write(f"H{to_subscript(current_layer-1)} = {H_n_1_r}")

st.latex(r"H_n = \sum_{i=1}^n h_i")
h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(current_layer)])
st.latex(r"H_n = " + h_terms_n)
st.write(f"H{to_subscript(current_layer)} = {H_n_r}")

st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")

if current_layer > 1:
    numerator = " + ".join([f"{round(E_values[i],3)} \cdot {round(h_values[i],3)}" for i in range(current_layer-1)])
    denominator = " + ".join([f"{round(h_values[i],3)}" for i in range(current_layer-1)])
    formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{round(weighted_sum_n_1,3)}}}{{{round(sum_h_n_1,3)}}} = {Esr_r}"
else:
    formula_with_values = r"Esr = 0 \quad \text{(няма пластове преди текущия)}"
st.latex(formula_with_values)

ratio_display = rf"\frac{{H_n}}{{D}} = \frac{{{H_n_r}}}{{{round(D,3)}}} = {ratio_r}"
st.latex(ratio_display)

En = E_values[current_layer-1]
En_r = round(En, 3)

st.markdown("### Изчисления с пласт")

st.latex(rf"E_{{{current_layer}}} = {En_r}")

Esr_over_En = Esr / En if En != 0 else 0
Esr_over_En_r = round(Esr_over_En, 3)
st.latex(rf"\frac{{Esr}}{{E_{{{current_layer}}}}} = \frac{{{Esr_r}}}{{{En_r}}} = {Esr_over_En_r}")

En_over_Ed = En / Ed if Ed != 0 else 0
En_over_Ed_r = round(En_over_Ed, 3)
st.latex(rf"\frac{{E_{{{current_layer}}}}}{{E_d}} = \frac{{{En_r}}}{{{Ed_r}}} = {En_over_Ed_r}")

# Зареждане на данни и построяване на графика
try:
    df_original = pd.read_csv("danni.csv")
    df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)
except FileNotFoundError:
    st.error("❌ Липсват нужните файлове dati.csv и/или Оразмеряване на опън за междиннен плстH_D.csv.")
    st.stop()

fig = go.Figure()

if 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {round(level,3)}',
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
            name=f'Esr/Ei = {round(sr_Ei,3)}',
            line=dict(width=2)
        ))

# --- Интерполация на y за Esr/Ei и Hn/D

sr_Ei_values = sorted(df_new['sr_Ei'].unique())
target_sr_Ei = Esr_over_En_r
target_Hn_D = ratio_r

y_at_ratio = None
interp_error = False

if target_sr_Ei < sr_Ei_values[0] or target_sr_Ei > sr_Ei_values[-1]:
    st.error(f"❌ Стойността Esr/Ei = {target_sr_Ei} е извън обхвата на наличните изолинии.")
    interp_error = True
else:
    if target_sr_Ei in sr_Ei_values:
        df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')
        y_at_ratio = np.interp(target_Hn_D, df_target['H/D'].values, df_target['y'].values)
    else:
        # Намерете индекси на съседни стойности
        lower_idx = None
        upper_idx = None
        for i in range(len(sr_Ei_values) - 1):
            if sr_Ei_values[i] <= target_sr_Ei <= sr_Ei_values[i + 1]:
                lower_idx = i
                upper_idx = i + 1
                break
        if lower_idx is None or upper_idx is None:
            st.error(f"❌ Не може да се намери интервал за Esr/Ei = {target_sr_Ei}")
            interp_error = True
        else:
            lower_val = sr_Ei_values[lower_idx]
            upper_val = sr_Ei_values[upper_idx]

            df_lower = df_new[df_new['sr_Ei'] == lower_val].sort_values(by='H/D')
            df_upper = df_new[df_new['sr_Ei'] == upper_val].sort_values(by='H/D')

            y_lower = np.interp(target_Hn_D, df_lower['H/D'].values, df_lower['y'].values)
            y_upper = np.interp(target_Hn_D, df_upper['H/D'].values, df_upper['y'].values)

            t = (target_sr_Ei - lower_val) / (upper_val - lower_val)
            y_at_ratio = y_lower + t * (y_upper - y_lower)

if not interp_error and y_at_ratio is not None:
    # Добавяне на вертикална линия на Hn/D
    fig.add_trace(go.Scatter(
        x=[ratio_r, ratio_r],
        y=[0, y_at_ratio],
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Вертикална линия на Hn/D'
    ))

    # Добавяне на червена точка на пресечната точка (интерполирана точка)
    fig.add_trace(go.Scatter(
        x=[ratio_r],
        y=[y_at_ratio],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Точка на пресичане'
    ))

    st.markdown(f"### Интерполирана стойност y за Esr/Ei = {target_sr_Ei} и Hn/D = {ratio_r}: {round(y_at_ratio,3)}")
else:
    st.markdown("Няма интерполирана стойност за текущите параметри.")

fig.update_layout(
    xaxis_title="Hn/D",
    yaxis_title="y",
    width=700,
    height=500,
    template="simple_white",
    title="Графика на опънното напрежение"
)

st.plotly_chart(fig)
