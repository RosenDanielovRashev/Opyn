import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междиен пласт от пътнатата конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Initialize session state
if 'layer_results' not in st.session_state:
    st.session_state.layer_results = {}

# Input parameters
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=4)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

# Input data for all layers
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

# Layer selection
st.markdown("### Избери пласт за проверка")
selected_layer = st.selectbox("Пласт за проверка", options=[f"Пласт {i+1}" for i in range(n)], index=n-1)
layer_idx = int(selected_layer.split()[-1]) - 1

# Calculation function
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

# Calculate button
if st.button(f"Изчисли за пласт {layer_idx+1}"):
    results = calculate_layer(layer_idx)
    st.success(f"Изчисленията за пласт {layer_idx+1} са запазени!")

# Display results
if layer_idx in st.session_state.layer_results:
    results = st.session_state.layer_results[layer_idx]
    
    st.markdown(f"### Резултати за пласт {layer_idx+1}")
    
    # Display formulas and calculations
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

    # Visualization
    try:
        df_original = pd.read_csv("danni.csv")
        df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
        df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

        fig = go.Figure()

        # Add isolines
        if 'Ei/Ed' in df_original.columns:
            for level in sorted(df_original['Ei/Ed'].unique()):
                df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'], y=df_level['y'],
                    mode='lines', name=f'Ei/Ed = {round(level,3)}',
                    line=dict(width=2)
                ))

        if 'sr_Ei' in df_new.columns:
            for sr_Ei in sorted(df_new['sr_Ei'].unique()):
                df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'], y=df_level['y'],
                    mode='lines', name=f'Esr/Ei = {round(sr_Ei,3)}',
                    line=dict(width=2)
                ))

        # Interpolation and marking points
        if layer_idx > 0:
            sr_Ei_values = sorted(df_new['sr_Ei'].unique())
            target_sr_Ei = results['Esr_over_En_r']
            target_Hn_D = results['ratio_r']

            y_at_ratio = None
            if min(sr_Ei_values) <= target_sr_Ei <= max(sr_Ei_values):
                if target_sr_Ei in sr_Ei_values:
                    df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')
                    y_at_ratio = np.interp(target_Hn_D, df_target['H/D'], df_target['y'])
                else:
                    for i in range(len(sr_Ei_values)-1):
                        if sr_Ei_values[i] < target_sr_Ei < sr_Ei_values[i+1]:
                            df_lower = df_new[df_new['sr_Ei'] == sr_Ei_values[i]].sort_values(by='H/D')
                            df_upper = df_new[df_new['sr_Ei'] == sr_Ei_values[i+1]].sort_values(by='H/D')
                            
                            y_lower = np.interp(target_Hn_D, df_lower['H/D'], df_lower['y'])
                            y_upper = np.interp(target_Hn_D, df_upper['H/D'], df_upper['y'])
                            
                            y_at_ratio = y_lower + (y_upper - y_lower) * (target_sr_Ei - sr_Ei_values[i]) / (sr_Ei_values[i+1] - sr_Ei_values[i])
                            break

            if y_at_ratio is not None:
                fig.add_trace(go.Scatter(
                    x=[target_Hn_D, target_Hn_D], y=[0, y_at_ratio],
                    mode='lines', line=dict(color='blue', dash='dash'),
                    name='Вертикална линия'
                ))

                fig.add_trace(go.Scatter(
                    x=[target_Hn_D], y=[y_at_ratio],
                    mode='markers', marker=dict(color='red', size=10),
                    name='Точка на интерполация'
                ))

                Ei_Ed_target = results['En_over_Ed_r']
                if 'Ei/Ed' in df_original.columns:
                    Ei_Ed_values = sorted(df_original['Ei/Ed'].unique())
                    if min(Ei_Ed_values) <= Ei_Ed_target <= max(Ei_Ed_values):
                        x_intercept = None
                        
                        if Ei_Ed_target in Ei_Ed_values:
                            df_level = df_original[df_original['Ei/Ed'] == Ei_Ed_target].sort_values(by='H/D')
                            x_intercept = np.interp(y_at_ratio, df_level['y'], df_level['H/D'])
                        else:
                            for i in range(len(Ei_Ed_values)-1):
                                if Ei_Ed_values[i] < Ei_Ed_target < Ei_Ed_values[i+1]:
                                    df_lower = df_original[df_original['Ei/Ed'] == Ei_Ed_values[i]].sort_values(by='H/D')
                                    df_upper = df_original[df_original['Ei/Ed'] == Ei_Ed_values[i+1]].sort_values(by='H/D')
                                    
                                    x_lower = np.interp(y_at_ratio, df_lower['y'], df_lower['H/D'])
                                    x_upper = np.interp(y_at_ratio, df_upper['y'], df_upper['H/D'])
                                    
                                    x_intercept = x_lower + (x_upper - x_lower) * (Ei_Ed_target - Ei_Ed_values[i]) / (Ei_Ed_values[i+1] - Ei_Ed_values[i])
                                    break

                        if x_intercept is not None:
                            fig.add_trace(go.Scatter(
                                x=[x_intercept], y=[y_at_ratio],
                                mode='markers', marker=dict(color='orange', size=12),
                                name='Пресечна точка'
                            ))

                            # Добавени линии между точките
                            fig.add_trace(go.Scatter(
                                x=[target_Hn_D, x_intercept],
                                y=[y_at_ratio, y_at_ratio],
                                mode='lines',
                                line=dict(color='green', dash='dot'),
                                name='Линия между червена и оранжева точка'
                            ))

                            fig.add_trace(go.Scatter(
                                x=[x_intercept, x_intercept],   # една и съща X, за вертикална линия
                                y=[y_at_ratio, 2.5],            # от текущото y до 2.5 по y
                                mode='lines',
                                line=dict(color='purple', dash='dash'),
                                name='Линия от оранжева точка до 2.5'
                            ))

                            sigma_r = round(x_intercept / 2, 3)
                            st.markdown(f"**σr за пласт {layer_idx+1} = {sigma_r}**")

        # Уверяваме се, че оста x обхваща поне 0 до 3 за видимост на линията до 2.5
        fig.update_layout(
            title=f"Графика за пласт {layer_idx+1}",
            xaxis_title="H/D",
            yaxis_title="y",
            legend_title="Легенда",
            height=600
        )
        fig.update_xaxes(range=[0, max(3, 2.5)])

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Грешка при зареждане на данните: {str(e)}")
