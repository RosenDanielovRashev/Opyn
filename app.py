import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междиен пласт от пътната конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

if 'step' not in st.session_state:
    st.session_state.step = 1

if 'calculation_1' not in st.session_state:
    st.session_state.calculation_1 = None
if 'calculation_2' not in st.session_state:
    st.session_state.calculation_2 = None

# Тук слагаме функцията за изчисление (същата, както по-горе; може да я рефакторираме за яснота)
def calculate_and_plot(n, D, h_values, E_values, Ed):
    # същата логика от предишния код за изчисления и графика
    h_array = np.array(h_values)
    E_array = np.array(E_values)

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

    Ed_r = round(Ed, 3)
    En = E_values[-1]
    En_r = round(En, 3)
    Esr_over_En = Esr / En if En != 0 else 0
    Esr_over_En_r = round(Esr_over_En, 3)
    En_over_Ed = En / Ed if Ed != 0 else 0
    En_over_Ed_r = round(En_over_Ed, 3)

    # Зареждане на данни и построяване на графика
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

    # Интерполация (същата като преди)
    sr_Ei_values = sorted(df_new['sr_Ei'].unique())
    target_sr_Ei = Esr_over_En_r
    target_Hn_D = ratio_r

    y_at_ratio = None
    interp_error = False

    if target_sr_Ei < sr_Ei_values[0] or target_sr_Ei > sr_Ei_values[-1]:
        interp_error = True
    else:
        if target_sr_Ei in sr_Ei_values:
            df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')
            y_at_ratio = np.interp(target_Hn_D, df_target['H/D'].values, df_target['y'].values)
        else:
            lower_idx = None
            upper_idx = None
            for i in range(len(sr_Ei_values) - 1):
                if sr_Ei_values[i] <= target_sr_Ei <= sr_Ei_values[i + 1]:
                    lower_idx = i
                    upper_idx = i + 1
                    break
            if lower_idx is None or upper_idx is None:
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
        fig.add_trace(go.Scatter(
            x=[ratio_r, ratio_r],
            y=[0, y_at_ratio],
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Вертикална линия на Hn/D'
        ))

        fig.add_trace(go.Scatter(
            x=[ratio_r],
            y=[y_at_ratio],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Интерполирана точка'
        ))

        def interp_x_for_y(df, y_target):
            x_arr = df['H/D'].values
            y_arr = df['y'].values
            for k in range(len(y_arr) - 1):
                y1, y2 = y_arr[k], y_arr[k + 1]
                if (y1 - y_target) * (y2 - y_target) <= 0:
                    x1, x2 = x_arr[k], x_arr[k + 1]
                    if y2 == y1:
                        return x1
                    t_local = (y_target - y1) / (y2 - y1)
                    x_interp = x1 + t_local * (x2 - x1)
                    return x_interp
            return None

        Ei_Ed_target = En_over_Ed_r
        Ei_Ed_values_sorted = sorted(df_original['Ei/Ed'].unique())
        lower_index_EiEd = None

        if Ei_Ed_target < Ei_Ed_values_sorted[0] or Ei_Ed_target > Ei_Ed_values_sorted[-1]:
            x_intercept = None
        else:
            for i in range(len(Ei_Ed_values_sorted) - 1):
                if Ei_Ed_values_sorted[i] <= Ei_Ed_target <= Ei_Ed_values_sorted[i + 1]:
                    lower_index_EiEd = i
                    break

            if Ei_Ed_target in Ei_Ed_values_sorted:
                df_level = df_original[df_original['Ei/Ed'] == Ei_Ed_target].sort_values(by='H/D')
                x_intercept = interp_x_for_y(df_level, y_at_ratio)
            elif lower_index_EiEd is not None:
                low_val = Ei_Ed_values_sorted[lower_index_EiEd]
                high_val = Ei_Ed_values_sorted[lower_index_EiEd + 1]

                df_low = df_original[df_original['Ei/Ed'] == low_val].sort_values(by='H/D')
                df_high = df_original[df_original['Ei/Ed'] == high_val].sort_values(by='H/D')

                x_low = interp_x_for_y(df_low, y_at_ratio)
                x_high = interp_x_for_y(df_high, y_at_ratio)

                if x_low is not None and x_high is not None:
                    t_EiEd = (Ei_Ed_target - low_val) / (high_val - low_val)
                    x_intercept = x_low + t_EiEd * (x_high - x_low)
                else:
                    x_intercept = None
            else:
                x_intercept = None

        if x_intercept is not None:
            fig.add_trace(go.Scatter(
                x=[x_intercept],
                y=[y_at_ratio],
                mode='markers',
                marker=dict(color='orange', size=12, symbol='circle'),
                name='Пресечна точка с Ei/Ed'
            ))

            fig.add_trace(go.Scatter(
                x=[x_intercept, ratio_r],
                y=[y_at_ratio, y_at_ratio],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                showlegend=False
            ))

            fig.add_trace(go.Scatter(
                x=[x_intercept, x_intercept],
                y=[y_at_ratio, 2.5],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                showlegend=False
            ))
        sigma_r = round(x_intercept / 2, 3) if x_intercept is not None else None
    else:
        sigma_r = None

    fig.update_layout(
        title="Графика",
        xaxis_title="H/D",
        yaxis_title="y",
        legend_title="Легенда"
    )

    results = {
        "H_n_1_r": H_n_1_r,
        "H_n_r": H_n_r,
        "Esr_r": Esr_r,
        "ratio_r": ratio_r,
        "Ed_r": Ed_r,
        "En_r": En_r,
        "Esr_over_En_r": Esr_over_En_r,
        "En_over_Ed_r": En_over_Ed_r,
        "fig": fig,
        "sigma_r": sigma_r,
        "interp_error": interp_error,
    }

    return results

def input_data(calc_num):
    st.markdown(f"## Въвеждане на данни за изчисление {calc_num}")

    n = st.number_input(f"Брой пластове (n) за изчисление {calc_num}", min_value=2, step=1, value=3, key=f"n_{calc_num}")
    D = st.selectbox(f"Избери D за изчисление {calc_num}", options=[32.04, 34.0], index=0, key=f"D_{calc_num}")

    st.markdown(f"### Въведи стойности за всеки пласт за изчисление {calc_num}")
    h_values = []
    E_values = []
    cols = st.columns(2)
    for i in range(n):
        with cols[0]:
            h = st.number_input(f"h{to_subscript(i+1)} за изчисление {calc_num}", value=4.0, step=0.1, key=f"h_{calc_num}_{i}")
            h_values.append(h)
        with cols[1]:
            E = st.number_input(f"E{to_subscript(i+1)} за изчисление {calc_num}", value=1000.0, step=0.1, key=f"E_{calc_num}_{i}")
            E_values.append(E)

    Ed = st.number_input(f"Ed за изчисление {calc_num}", value=100.0, step=0.1, key=f"Ed_{calc_num}")

    return n, D, h_values, E_values, Ed

# Стъпка 1: Първо изчисление
if st.session_state.step == 1:
    n, D, h_values, E_values, Ed = input_data(1)
    if st.button("Изчисли първо изчисление"):
        res = calculate_and_plot(n, D, h_values, E_values, Ed)
        st.session_state.calculation_1 = res
        st.session_state.step = 2  # преминаваме към стъпка 2

    if st.session_state.calculation_1:
        st.markdown("### Резултати от първото изчисление")
        res = st.session_state.calculation_1
        st.write(f"Hₙ₋₁ = {res['H_n_1_r']}")
        st.write(f"Hₙ = {res['H_n_r']}")
        st.write(f"Esr = {res['Esr_r']}")
        st.write(f"Hn/D = {res['ratio_r']}")
        st.write(f"Ed = {res['Ed_r']}")
        st.write(f"En = {res['En_r']}")
        st.write(f"Esr/En = {res['Esr_over_En_r']}")
        st.write(f"En/Ed = {res['En_over_Ed_r']}")
        st.plotly_chart(res['fig'])
        if res['sigma_r'] is not None:
            st.markdown(f"**σr = {res['sigma_r']}**")
        else:
            st.markdown("**σr = -** (Няма изчислена стойност)")

    if st.session_state.calculation_1:
        if st.button("Напред към второ изчисление"):
            st.session_state.step = 2

# Стъпка 2: Второ изчисление
elif st.session_state.step == 2:
    n, D, h_values, E_values, Ed = input_data(2)
    if st.button("Изчисли второ изчисление"):
        res = calculate_and_plot(n, D, h_values, E_values, Ed)
        st.session_state.calculation_2 = res

    if st.session_state.calculation_2:
        st.markdown("### Резултати от второто изчисление")
        res = st.session_state.calculation_2
        st.write(f"Hₙ₋₁ = {res['H_n_1_r']}")
        st.write(f"Hₙ = {res['H_n_r']}")
        st.write(f"Esr = {res['Esr_r']}")
        st.write(f"Hn/D = {res['ratio_r']}")
        st.write(f"Ed = {res['Ed_r']}")
        st.write(f"En = {res['En_r']}")
        st.write(f"Esr/En = {res['Esr_over_En_r']}")
        st.write(f"En/Ed = {res['En_over_Ed_r']}")
        st.plotly_chart(res['fig'])
        if res['sigma_r'] is not None:
            st.markdown(f"**σr = {res['sigma_r']}**")
        else:
            st.markdown("**σr = -** (Няма изчислена стойност)")

    if st.button("Обратно към първо изчисление"):
        st.session_state.step = 1
