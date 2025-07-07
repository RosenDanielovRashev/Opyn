import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Определяне опънното напрежение в междинен пласт от пътната конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

if "results" not in st.session_state:
    st.session_state["results"] = []
if "current_result_idx" not in st.session_state:
    st.session_state["current_result_idx"] = -1

def calculate_for_layer(layer_idx):
    if layer_idx < 0 or layer_idx >= n:
        return None
    
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

    En = E_array[layer_idx]
    En_r = round(En, 3)
    
    Ed_r = round(Ed, 3)
    
    Esr_over_En = Esr / En if En != 0 else 0
    Esr_over_En_r = round(Esr_over_En, 3)
    
    En_over_Ed = En / Ed if Ed != 0 else 0
    En_over_Ed_r = round(En_over_Ed, 3)

    # Тук зареждам csv файлове и правя интерполации както преди
    df_original = pd.read_csv("danni.csv")
    df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)
    
    sr_Ei_values = sorted(df_new['sr_Ei'].unique())
    target_sr_Ei = Esr_over_En_r
    target_Hn_D = ratio_r

    y_at_ratio = None

    if target_sr_Ei < sr_Ei_values[0] or target_sr_Ei > sr_Ei_values[-1]:
        y_at_ratio = None
    else:
        lower_val, upper_val = None, None
        for i in range(len(sr_Ei_values) - 1):
            if sr_Ei_values[i] <= target_sr_Ei <= sr_Ei_values[i + 1]:
                lower_val = sr_Ei_values[i]
                upper_val = sr_Ei_values[i + 1]
                break

        if lower_val is not None and upper_val is not None:
            df_lower = df_new[df_new['sr_Ei'] == lower_val].sort_values(by='H/D')
            df_upper = df_new[df_new['sr_Ei'] == upper_val].sort_values(by='H/D')
            y_lower = np.interp(target_Hn_D, df_lower['H/D'], df_lower['y'])
            y_upper = np.interp(target_Hn_D, df_upper['H/D'], df_upper['y'])
            t = (target_sr_Ei - lower_val) / (upper_val - lower_val)
            y_at_ratio = y_lower + t * (y_upper - y_lower)
        else:
            y_at_ratio = None
    
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
    
    x_intercept = None
    if y_at_ratio is not None:
        Ei_Ed_target = En_over_Ed_r
        Ei_Ed_values_sorted = sorted(df_original['Ei/Ed'].unique())
        lower_index_EiEd = None

        if Ei_Ed_target >= Ei_Ed_values_sorted[0] and Ei_Ed_target <= Ei_Ed_values_sorted[-1]:
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
    
    sigma_r = round(x_intercept / 2, 3) if x_intercept is not None else None

    return {
        "пласт": layer_idx + 1,
        "H_n_1": H_n_1_r,
        "H_n": H_n_r,
        "Esr": Esr_r,
        "Hn/D": ratio_r,
        "Ed": Ed_r,
        "En": En_r,
        "Esr/En": Esr_over_En_r,
        "En/Ed": En_over_Ed_r,
        "y_at_ratio": y_at_ratio,
        "x_intercept": x_intercept,
        "σr": sigma_r,
        "h_values": h_values,
        "E_values": E_values,
        "D": D
    }

# Входни данни
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[32.04, 34.0, 33.0], index=0)

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

h_array = np.array(h_values)
E_array = np.array(E_values)

Ed = st.number_input("Ed", value=100.0, step=0.1)

def prev_result():
    if st.session_state["current_result_idx"] > 0:
        st.session_state["current_result_idx"] -= 1

def next_result():
    if st.session_state["current_result_idx"] < len(st.session_state["results"]) - 1:
        st.session_state["current_result_idx"] += 1

col_nav1, col_nav2 = st.columns([1,1])
with col_nav1:
    st.button("← Предишен", on_click=prev_result)
with col_nav2:
    st.button("Следващ →", on_click=next_result)

if st.button(f"Изчисли опън за пласт {n}"):
    res = calculate_for_layer(n-1)
    if res is not None:
        st.session_state["results"].append(res)
        st.session_state["current_result_idx"] = len(st.session_state["results"]) - 1
        st.success(f"Добавено изчисление за пласт {n}")

if n > 1:
    if st.button(f"Изчисли опън за съседен пласт {n-1}"):
        res = calculate_for_layer(n-2)
        if res is not None:
            st.session_state["results"].append(res)
            st.session_state["current_result_idx"] = len(st.session_state["results"]) - 1
            st.success(f"Добавено изчисление за пласт {n-1}")

if st.session_state["results"] and st.session_state["current_result_idx"] >= 0:
    current = st.session_state["results"][st.session_state["current_result_idx"]]
    st.markdown(f"### Изчисление #{st.session_state['current_result_idx'] + 1} (Пласт {current['пласт']})")
    st.write(f"Hₙ₋₁ = {current['H_n_1']}")
    st.write(f"Hₙ = {current['H_n']}")
    st.write(f"Esr = {current['Esr']}")
    st.write(f"Hₙ/D = {current['Hn/D']}")
    st.write(f"Ed = {current['Ed']}")
    st.write(f"Eₙ = {current['En']}")
    st.write(f"Esr/Eₙ = {current['Esr/En']}")
    st.write(f"Eₙ/Ed = {current['En/Ed']}")
    st.write(f"y при Hₙ/D = {current['y_at_ratio']}")
    st.write(f"x пресечна точка = {current['x_intercept']}")
    st.markdown(f"**σr = {current['σr'] if current['σr'] is not None else '-'}**")

    # Графики
    st.markdown("### Графики")

    # Графика на височини на пластовете
    fig_h = go.Figure()
    fig_h.add_trace(go.Bar(x=[f"h{to_subscript(i+1)}" for i in range(n)], y=current['h_values'], name="h (дебелина)"))
    fig_h.update_layout(title="Дебелина на пластовете (h)", yaxis_title="h [cm]")
    st.plotly_chart(fig_h, use_container_width=True)

    # Графика на модули на пластовете
    fig_E = go.Figure()
    fig_E.add_trace(go.Bar(x=[f"E{to_subscript(i+1)}" for i in range(n)], y=current['E_values'], name="E (модул на еластичност)"))
    fig_E.update_layout(title="Модул на еластичност на пластовете (E)", yaxis_title="E [MPa]")
    st.plotly_chart(fig_E, use_container_width=True)

    # Ако има σr, показваме го като линия за избрания пласт
    if current['σr'] is not None:
        fig_sigma = go.Figure()
        fig_sigma.add_trace(go.Bar(x=[f"Пласт {current['пласт']}"], y=[current['σr']], name="σr (опън)"))
        fig_sigma.update_layout(title="Опънно напрежение σr за пласт", yaxis_title="σr [MPa]", yaxis_range=[0, max(10, current['σr']*1.2)])
        st.plotly_chart(fig_sigma, use_container_width=True)

if st.button("Изчисти всички резултати"):
    st.session_state["results"] = []
    st.session_state["current_result_idx"] = -1
    st.experimental_rerun()
