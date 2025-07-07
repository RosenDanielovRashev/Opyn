import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междиен пласт от пътнатата конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Инициализация на session state
if 'layer_results' not in st.session_state:
    st.session_state.layer_results = {}

# Входни параметри
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=4)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

# Въвеждане на данни за всички пластове
st.markdown("### Въведи стойности за всички пластове")
h_values = []
E_values = []
Ed_values = []

cols = st.columns(3)
for i in range(n):
    with cols[0]:
        h = st.number_input(f"h{to_subscript(i+1)}", value=4.0, step=0.1, key=f"h_{i}")
        h_values.append(h)
    with cols[1]:
        default_E = [1200.0, 1000.0, 800.0, 400.0][i] if i < 4 else 400.0
        E = st.number_input(f"E{to_subscript(i+1)}", value=default_E, step=0.1, key=f"E_{i}")
        E_values.append(E)
    with cols[2]:
        Ed = st.number_input(f"Ed{to_subscript(i+1)}", value=30.0, step=0.1, key=f"Ed_{i}")
        Ed_values.append(Ed)

# Избор на пласт за проверка
st.markdown("### Избери пласт за проверка")
selected_layer = st.selectbox("Пласт за проверка", options=[f"Пласт {i+1}" for i in range(n)], index=n-1)
layer_idx = int(selected_layer.split()[-1]) - 1

# Функция за изчисления за конкретен пласт
def calculate_layer(layer_index):
    h_array = np.array(h_values[:layer_index+1])
    E_array = np.array(E_values[:layer_index+1])
    current_Ed = Ed_values[layer_index]
    
    sum_h_n_1 = h_array[:-1].sum() if layer_index > 0 else 0
    weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1]) if layer_index > 0 else 0
    Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0
    
    H_n = h_array.sum()
    H_n_1 = sum_h_n_1
    
    results = {
        'H_n_1_r': round(H_n_1, 3),
        'H_n_r': round(H_n, 3),
        'Esr_r': round(Esr, 3),
        'ratio_r': round(H_n / D, 3) if D != 0 else 0,
        'En_r': round(E_values[layer_index], 3),
        'Ed_r': round(current_Ed, 3),
        'Esr_over_En_r': round(Esr / E_values[layer_index], 3) if E_values[layer_index] != 0 else 0,
        'En_over_Ed_r': round(E_values[layer_index] / current_Ed, 3) if current_Ed != 0 else 0,
        'h_values': h_values.copy(),
        'E_values': E_values.copy(),
        'n_for_calc': layer_index + 1
    }
    
    st.session_state.layer_results[layer_index] = results
    return results

# Бутон за изчисление на текущия пласт
if st.button(f"Изчисли за пласт {layer_idx+1}"):
    results = calculate_layer(layer_idx)
    st.success(f"Изчисленията за пласт {layer_idx+1} са запазени!")

# Показване на резултатите за избрания пласт
if layer_idx in st.session_state.layer_results:
    results = st.session_state.layer_results[layer_idx]
    
    st.markdown(f"### Резултати за пласт {layer_idx+1}")
    
    # Показване на формули и изчисления
    st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
    if layer_idx > 0:
        h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(layer_idx)])
        st.latex(r"H_{n-1} = " + h_terms)
    st.write(f"H{to_subscript(layer_idx)} = {results['H_n_1_r']}")

    st.latex(r"H_n = \sum_{i=1}^n h_i")
    h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(results['n_for_calc'])])
    st.latex(r"H_n = " + h_terms_n)
    st.write(f"H{to_subscript(results['n_for_calc'])} = {results['H_n_r']}")

    if layer_idx > 0:
        st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")
        numerator = " + ".join([f"{results['E_values'][i]} \cdot {results['h_values'][i]}" for i in range(layer_idx)])
        denominator = " + ".join([f"{results['h_values'][i]}" for i in range(layer_idx)])
        st.latex(fr"Esr = \frac{{{numerator}}}{{{denominator}}} = {results['Esr_r']}")
    else:
        st.write("Esr = 0 (няма предишни пластове)")

    st.latex(fr"\frac{{H_n}}{{D}} = \frac{{{results['H_n_r']}}}{{{D}}} = {results['ratio_r']}")
    st.latex(fr"E_{{{layer_idx+1}}} = {results['En_r']}")
    st.latex(fr"\frac{{Esr}}{{E_{{{layer_idx+1}}}}} = {results['Esr_over_En_r']}")
    st.latex(fr"\frac{{E_{{{layer_idx+1}}}}}{{Ed_{{{layer_idx+1}}}}} = \frac{{{results['En_r']}}}{{{results['Ed_r']}}} = {results['En_over_Ed_r']}")

    # Визуализация
    try:
        # Създаване на фигура
        fig = go.Figure()

        # Добавяне на изолинии (примерни данни)
        for i in range(1, 6):
            fig.add_trace(go.Scatter(
                x=[0, 1, 2, 3],
                y=[i*0.5, i*0.5+0.2, i*0.5+0.4, i*0.5+0.3],
                mode='lines',
                name=f'Изолиния {i}',
                line=dict(width=1)
            ))

        # Ако имаме резултати за интерполация
        if layer_idx > 0 and 'ratio_r' in results and 'Esr_over_En_r' in results:
            target_Hn_D = results['ratio_r']
            y_at_ratio = target_Hn_D * 0.8  # Примерна стойност за демонстрация
            
            # Вертикална линия
            fig.add_trace(go.Scatter(
                x=[target_Hn_D, target_Hn_D],
                y=[0, y_at_ratio],
                mode='lines',
                line=dict(color='blue', dash='dash', width=2),
                name=f'H/D = {target_Hn_D}'
            ))
            
            # Хоризонтална линия
            fig.add_trace(go.Scatter(
                x=[0, target_Hn_D],
                y=[y_at_ratio, y_at_ratio],
                mode='lines',
                line=dict(color='green', dash='dot', width=1),
                name=f'y = {round(y_at_ratio,2)}'
            ))
            
            # Основна точка
            fig.add_trace(go.Scatter(
                x=[target_Hn_D],
                y=[y_at_ratio],
                mode='markers+text',
                marker=dict(color='red', size=12),
                name='Точка',
                text=[f'({target_Hn_D}, {round(y_at_ratio,2)})'],
                textposition="top right"
            ))

            # Примерна пресечна точка
            x_intercept = target_Hn_D * 0.7
            if x_intercept is not None:
                # Линия между точките
                fig.add_trace(go.Scatter(
                    x=[x_intercept, target_Hn_D],
                    y=[y_at_ratio, y_at_ratio],
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dash'),
                    name='Връзка'
                ))
                
                # Вертикална линия от пресечната точка
                fig.add_trace(go.Scatter(
                    x=[x_intercept, x_intercept],
                    y=[y_at_ratio, 2.5],
                    mode='lines',
                    line=dict(color='purple', dash='dot', width=1),
                    name='σr линия'
                ))
                
                # Пресечна точка
                fig.add_trace(go.Scatter(
                    x=[x_intercept],
                    y=[y_at_ratio],
                    mode='markers+text',
                    marker=dict(color='orange', size=12, symbol='diamond'),
                    name='Пресечна точка',
                    text=[f'({round(x_intercept,2)}, {round(y_at_ratio,2)})'],
                    textposition="bottom center"
                ))

                # Изчисление на σr
                sigma_r = round(x_intercept / 2, 3)
                st.markdown(f"**σr за пласт {layer_idx+1} = {sigma_r}**")

        # Конфигурация на графиката
        fig.update_layout(
            title=f"Графика за пласт {layer_idx+1}",
            xaxis_title="H/D",
            yaxis_title="y",
            legend_title="Легенда",
            height=600,
            xaxis=dict(range=[0, 3]),
            yaxis=dict(range=[0, 2.5])
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Грешка при визуализация: {str(e)}")
