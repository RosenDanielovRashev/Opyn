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

# Layer selection (без първите два пласта)
st.markdown("### Избери пласт за проверка")

available_layers = [f"Пласт {i+1}" for i in range(2, n)]  # започваме от пласт 3 (индекс 2)

if len(available_layers) == 0:
    st.warning("Няма достатъчно пластове за избор (трябва поне 3).")
    selected_layer = None
    layer_idx = None
else:
    selected_layer = st.selectbox("Пласт за проверка", options=available_layers, index=len(available_layers)-1)
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
        'Ed_r': round(current_Ed, 3),  # Store Ed value for this layer
        'Esr_over_En_r': round(Esr / E_values[layer_index], 3) if E_values[layer_index] != 0 else 0,
        'En_over_Ed_r': round(E_values[layer_index] / current_Ed, 3) if current_Ed != 0 else 0,
        'h_values': h_values.copy(),
        'E_values': E_values.copy(),
        'n_for_calc': layer_index + 1
    }
    
    st.session_state.layer_results[layer_index] = results
    return results

# Calculate button
if layer_idx is not None:
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
            if layer_idx > 0:  # Only if there are previous layers
                sr_Ei_values = sorted(df_new['sr_Ei'].unique())
                target_sr_Ei = results['Esr_over_En_r']
                target_Hn_D = results['ratio_r']

                if min(sr_Ei_values) <= target_sr_Ei <= max(sr_Ei_values):
                    # Find y coordinate
                    y_at_ratio = None
                    if target_sr_Ei in sr_Ei_values:
                        df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')
                        y_at_ratio = np.interp(target_Hn_D, df_target['H/D'], df_target['y'])
                    else:
                        for i in range(len(sr_Ei_values)-1):
                            if sr_Ei_values[i] < target_sr_Ei < sr_Ei_values[i+1]:
                                df_low = df_new[df_new['sr_Ei'] == sr_Ei_values[i]].sort_values(by='H/D')
                                df_high = df_new[df_new['sr_Ei'] == sr_Ei_values[i+1]].sort_values(by='H/D')
                                y_low = np.interp(target_Hn_D, df_low['H/D'], df_low['y'])
                                y_high = np.interp(target_Hn_D, df_high['H/D'], df_high['y'])
                                y_at_ratio = y_low + (y_high - y_low) * (target_sr_Ei - sr_Ei_values[i]) / (sr_Ei_values[i+1] - sr_Ei_values[i])
                                break

                    if y_at_ratio is not None:
                        # Mark points on plot
                        fig.add_trace(go.Scatter(
                            x=[target_Hn_D], y=[0],
                            mode='markers',
                            marker=dict(color='orange', size=12),
                            name='Точка на 0'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[target_Hn_D], y=[y_at_ratio],
                            mode='markers',
                            marker=dict(color='red', size=12),
                            name='Точка на y'
                        ))
                        # Линия свързваща двете точки (вертикална)
                        fig.add_trace(go.Scatter(
                            x=[target_Hn_D, target_Hn_D],
                            y=[0, y_at_ratio],
                            mode='lines',
                            line=dict(color='black', width=2),
                            showlegend=False
                        ))

            # --- Добавяне на невидим trace за втората ос (за да се покаже мащабът)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[None, None],  # y не влияе
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip',
                xaxis='x2'  # Свързваме с втората ос
            ))

            fig.update_layout(
                title='Графика на изолинии',
                xaxis=dict(
                    title='H/D',
                    showgrid=True,
                    zeroline=False,
                ),
                xaxis2=dict(
                    overlaying='x',
                    side='top',
                    range=[fig.layout.xaxis.range[0] if fig.layout.xaxis.range else None, 1],
                    showgrid=False,
                    zeroline=False,
                    tickvals=[0, 0.25, 0.5, 0.75, 1],
                    ticktext=['0', '0.25', '0.5', '0.75', '1'],
                    title='σr'
                ),
                yaxis=dict(
                    title='y',
                ),
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Възникна грешка при визуализацията: {e}")

else:
    st.info("Моля, добави поне 3 пласта, за да можеш да избереш пласт за проверка.")
