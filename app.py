import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междинен пласт от пътната конструкция (фиг.9.3)")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Входни параметри (винаги видими)
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

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

calculate = st.button("Изчисли")

if calculate:
    h_array = np.array(h_values)
    E_array = np.array(E_values)

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
    Ed_r = round(Ed, 3)
    En = E_values[-1]
    En_r = round(En, 3)

    # Показване на формули и резултати
    st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
    h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n-1)])
    st.latex(r"H_{n-1} = " + h_terms)
    st.write(f"H{to_subscript(n-1)} = {H_n_1_r}")

    st.latex(r"H_n = \sum_{i=1}^n h_i")
    h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n)])
    st.latex(r"H_n = " + h_terms_n)
    st.write(f"H{to_subscript(n)} = {H_n_r}")

    st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")
    numerator = " + ".join([f"{round(E_values[i],3)} \cdot {round(h_values[i],3)}" for i in range(n-1)])
    denominator = " + ".join([f"{round(h_values[i],3)}" for i in range(n-1)])
    st.latex(rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{round(weighted_sum_n_1,3)}}}{{{round(sum_h_n_1,3)}}} = {Esr_r}")

    st.latex(rf"\frac{{H_n}}{{D}} = \frac{{{H_n_r}}}{{{round(D,3)}}} = {ratio_r}")

    st.markdown("### Изчисления с последен пласт")
    st.latex(rf"E_{{{n}}} = {En_r}")
    Esr_over_En = Esr / En if En != 0 else 0
    Esr_over_En_r = round(Esr_over_En, 3)
    st.latex(rf"\frac{{Esr}}{{E_{{{n}}}}} = \frac{{{Esr_r}}}{{{En_r}}} = {Esr_over_En_r}")

    En_over_Ed = En / Ed if Ed != 0 else 0
    En_over_Ed_r = round(En_over_Ed, 3)
    st.latex(rf"\frac{{E_{{{n}}}}}{{E_d}} = \frac{{{En_r}}}{{{Ed_r}}} = {En_over_Ed_r}")

    # Зареждане на данни и графика
    df_original = pd.read_csv("danni.csv")
    df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

    fig = go.Figure()

    if 'Ei/Ed' in df_original.columns:
        for level in sorted(df_original['Ei/Ed'].unique()):
            df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
            fig.add_trace(go.Scatter(
                x=df_level['H/D'],
                y=df_level['y'],
                mode='lines',
                name=f'Ei/Ed = {round(level,3)}',
                line=dict(width=2)
            ))

    if 'sr_Ei' in df_new.columns:
        for sr_Ei in sorted(df_new['sr_Ei'].unique()):
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
            for i in range(len(sr_Ei_values) - 1):
                if sr_Ei_values[i] <= target_sr_Ei <= sr_Ei_values[i + 1]:
                    lower_idx, upper_idx = i, i + 1
                    break
            df_lower = df_new[df_new['sr_Ei'] == sr_Ei_values[lower_idx]].sort_values(by='H/D')
            df_upper = df_new[df_new['sr_Ei'] == sr_Ei_values[upper_idx]].sort_values(by='H/D')

            y_lower = np.interp(target_Hn_D, df_lower['H/D'].values, df_lower['y'].values)
            y_upper = np.interp(target_Hn_D, df_upper['H/D'].values, df_upper['y'].values)

            t = (target_sr_Ei - sr_Ei_values[lower_idx]) / (sr_Ei_values[upper_idx] - sr_Ei_values[lower_idx])
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
                    return x1 + t_local * (x2 - x1)
            return None

        Ei_Ed_target = En_over_Ed_r
        Ei_Ed_values_sorted = sorted(df_original['Ei/Ed'].unique())

        if Ei_Ed_target < Ei_Ed_values_sorted[0] or Ei_Ed_target > Ei_Ed_values_sorted[-1]:
            st.error(f"❌ Стойността Ei/Ed = {Ei_Ed_target} е извън обхвата на наличните изолинии.")
        else:
            lower_index_EiEd = None
            for i in range(len(Ei_Ed_values_sorted) - 1):
                if Ei_Ed_values_sorted[i] <= Ei_Ed_target <= Ei_Ed_values_sorted[i + 1]:
                    lower_index_EiEd = i
                    break

            if Ei_Ed_target in Ei_Ed_values_sorted:
                df_level = df_original[df_original['Ei/Ed'] == Ei_Ed_target].sort_values(by='H/D')
                x_intercept = interp_x_for_y(df_level, y_at_ratio)
            elif lower_index_EiEd is not None:
                df_low = df_original[df_original['Ei/Ed'] == Ei_Ed_values_sorted[lower_index_EiEd]].sort_values(by='H/D')
                df_high = df_original[df_original['Ei/Ed'] == Ei_Ed_values_sorted[lower_index_EiEd + 1]].sort_values(by='H/D')

                x_low = interp_x_for_y(df_low, y_at_ratio)
                x_high = interp_x_for_y(df_high, y_at_ratio)

                if x_low is not None and x_high is not None:
                    t_EiEd = (Ei_Ed_target - Ei_Ed_values_sorted[lower_index_EiEd]) / (Ei_Ed_values_sorted[lower_index_EiEd + 1] - Ei_Ed_values_sorted[lower_index_EiEd])
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
                st.markdown(f"### Пресечна стойност на H/D за Ei/Ed = {Ei_Ed_target}: {round(x_intercept, 3)}")

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Натиснете бутона „Изчисли“, за да видите резултатите.")
