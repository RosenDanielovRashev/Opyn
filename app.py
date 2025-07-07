import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междиен пласт от пътнатата конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Инициализация на session state за запазване на резултатите
if 'layer_results' not in st.session_state:
    st.session_state.layer_results = {}

# Входни параметри
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=4)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

# Въвеждане на h_i и E_i за всеки пласт
st.markdown("### Въведи стойности за всички пластове")
h_values = []
E_values = []

cols = st.columns(2)
for i in range(n):
    with cols[0]:
        h = st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        default_E = [1200.0, 1000.0, 800.0, 400.0][i] if i < 4 else 400.0
        E = st.number_input(f"E{to_subscript(i+1)}", value=default_E, step=0.1, key=f"E_{i}")
        E_values.append(E)

Ed = st.number_input("Ed", value=30.0, step=0.1)
Ed_r = round(Ed, 3)

# Избор на пласт за проверка
st.markdown("### Избери пласт за проверка")
selected_layer = st.selectbox("Пласт за проверка", options=[f"Пласт {i+1}" for i in range(n)], index=n-1)
layer_idx = int(selected_layer.split()[-1]) - 1

# Функция за изчисления
def calculate_layer(layer_index):
    h_array = np.array(h_values)
    E_array = np.array(E_values)
    
    n_for_calc = layer_index + 1
    sum_h_n_1 = h_array[:layer_index].sum()
    weighted_sum_n_1 = np.sum(E_array[:layer_index] * h_array[:layer_index])
    Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0
    
    H_n = h_array[:n_for_calc].sum()
    H_n_1 = sum_h_n_1
    
    H_n_1_r = round(H_n_1, 3)
    H_n_r = round(H_n, 3)
    Esr_r = round(Esr, 3)
    ratio = H_n / D if D != 0 else 0
    ratio_r = round(ratio, 3)
    
    En = E_values[layer_index]
    En_r = round(En, 3)
    Esr_over_En = Esr / En if En != 0 else 0
    Esr_over_En_r = round(Esr_over_En, 3)
    En_over_Ed = En / Ed if Ed != 0 else 0
    En_over_Ed_r = round(En_over_Ed, 3)
    
    # Запазване на резултатите
    st.session_state.layer_results[layer_index] = {
        'H_n_1_r': H_n_1_r,
        'H_n_r': H_n_r,
        'Esr_r': Esr_r,
        'ratio_r': ratio_r,
        'En_r': En_r,
        'Esr_over_En_r': Esr_over_En_r,
        'En_over_Ed_r': En_over_Ed_r,
        'h_values': h_values.copy(),
        'E_values': E_values.copy(),
        'n_for_calc': n_for_calc
    }

# Бутон за изчисления и визуализация
if st.button("Изчисли за всички пластове"):
    for i in range(n):
        calculate_layer(i)
    st.success(f"Изчисленията за всички {n} пласта са запазени!")

# Проверка дали има запазени резултати за избрания пласт
if layer_idx in st.session_state.layer_results:
    results = st.session_state.layer_results[layer_idx]
    
    st.markdown(f"### Резултати за пласт {layer_idx+1}")
    
    st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
    h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(layer_idx)])
    st.latex(r"H_{n-1} = " + h_terms)
    st.write(f"H{to_subscript(layer_idx)} = {results['H_n_1_r']}")

    st.latex(r"H_n = \sum_{i=1}^n h_i")
    h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(results['n_for_calc'])])
    st.latex(r"H_n = " + h_terms_n)
    st.write(f"H{to_subscript(results['n_for_calc'])} = {results['H_n_r']}")

    st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")
    
    numerator = " + ".join([f"{round(st.session_state.layer_results[layer_idx]['E_values'][i],3)} \cdot {round(st.session_state.layer_results[layer_idx]['h_values'][i],3)}" 
                          for i in range(layer_idx)])
    denominator = " + ".join([f"{round(st.session_state.layer_results[layer_idx]['h_values'][i],3)}" 
                           for i in range(layer_idx)])
    formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = {results['Esr_r']}"
    st.latex(formula_with_values)

    ratio_display = rf"\frac{{H_n}}{{D}} = \frac{{{results['H_n_r']}}}{{{round(D,3)}}} = {results['ratio_r']}"
    st.latex(ratio_display)

    st.markdown(f"### Изчисления с избран пласт {layer_idx+1}")

    st.latex(r"E_{" + str(results['n_for_calc']) + r"} = " + f"{results['En_r']}")

    st.latex(r"\frac{Esr}{E_{" + str(results['n_for_calc']) + r"}} = " + f"{results['Esr_over_En_r']}")
    st.latex(r"\frac{E_{" + str(results['n_for_calc']) + r"}}{E_d} = " + f"{results['En_over_Ed_r']}")

    # Визуализация на графиката
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

    # Интерполация
    sr_Ei_values = sorted(df_new['sr_Ei'].unique())
    target_sr_Ei = results['Esr_over_En_r']
    target_Hn_D = results['ratio_r']

    y_at_ratio = None
    if target_sr_Ei >= sr_Ei_values[0] and target_sr_Ei <= sr_Ei_values[-1]:
        if target_sr_Ei in sr_Ei_values:
            df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')
            y_at_ratio = np.interp(target_Hn_D, df_target['H/D'].values, df_target['y'].values)
        else:
            for i in range(len(sr_Ei_values) - 1):
                if sr_Ei_values[i] <= target_sr_Ei <= sr_Ei_values[i + 1]:
                    df_lower = df_new[df_new['sr_Ei'] == sr_Ei_values[i]].sort_values(by='H/D')
                    df_upper = df_new[df_new['sr_Ei'] == sr_Ei_values[i + 1]].sort_values(by='H/D')
                    
                    y_lower = np.interp(target_Hn_D, df_lower['H/D'].values, df_lower['y'].values)
                    y_upper = np.interp(target_Hn_D, df_upper['H/D'].values, df_upper['y'].values)
                    
                    t = (target_sr_Ei - sr_Ei_values[i]) / (sr_Ei_values[i + 1] - sr_Ei_values[i])
                    y_at_ratio = y_lower + t * (y_upper - y_lower)
                    break

    if y_at_ratio is not None:
        fig.add_trace(go.Scatter(
            x=[target_Hn_D, target_Hn_D],
            y=[0, y_at_ratio],
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Вертикална линия'
        ))

        fig.add_trace(go.Scatter(
            x=[target_Hn_D],
            y=[y_at_ratio],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Точка на интерполация'
        ))

    fig.update_layout(
        title=f"Графика за пласт {layer_idx+1}",
        xaxis_title="H/D",
        yaxis_title="y",
        legend_title="Легенда"
    )

    st.plotly_chart(fig)

elif st.button("Покажи графиката"):
    calculate_layer(layer_idx)
    st.rerun()
