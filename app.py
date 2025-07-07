import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Определяне опънното напрежение в междинен пласт от пътната конструкция фиг.9.3")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Инициализация на session_state
if 'current_layer' not in st.session_state:
    st.session_state.current_layer = 0

if 'n' not in st.session_state:
    st.session_state.n = 3  # начална стойност на броя пластове

if 'h_values' not in st.session_state:
    st.session_state.h_values = [4.0] * st.session_state.n

if 'E_values' not in st.session_state:
    st.session_state.E_values = [1000.0] * st.session_state.n

if 'Ed_values' not in st.session_state:
    st.session_state.Ed_values = [100.0] * st.session_state.n

# Въвеждане на брой пластове - само при промяна се променят размерите на списъците
n_input = st.number_input("Брой пластове (n)", min_value=2, max_value=10, step=1, value=st.session_state.n)

if n_input != st.session_state.n:
    old_n = st.session_state.n
    st.session_state.n = n_input

    if n_input > old_n:
        # Добавяме нови стойности с defaults
        st.session_state.h_values.extend([4.0] * (n_input - old_n))
        st.session_state.E_values.extend([1000.0] * (n_input - old_n))
        st.session_state.Ed_values.extend([100.0] * (n_input - old_n))
    elif n_input < old_n:
        # Скъсяваме списъците
        st.session_state.h_values = st.session_state.h_values[:n_input]
        st.session_state.E_values = st.session_state.E_values[:n_input]
        st.session_state.Ed_values = st.session_state.Ed_values[:n_input]

    # Ако current_layer е извън новия обхват, коригираме
    if st.session_state.current_layer >= n_input:
        st.session_state.current_layer = n_input - 1

n = st.session_state.n

# Избор на D (функционално е същото)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

st.write(f"Текущ пласт: {st.session_state.current_layer + 1} от {n}")

# Навигационни бутони
col1, col2 = st.columns(2)
with col1:
    if st.button("Назад"):
        if st.session_state.current_layer > 0:
            st.session_state.current_layer -= 1
with col2:
    if st.button("Напред"):
        if st.session_state.current_layer < n - 1:
            st.session_state.current_layer += 1

cur = st.session_state.current_layer

# Въвеждаме h и E за текущия пласт, стойностите се обновяват в сесията
h = st.number_input(f"h{to_subscript(cur+1)}", value=st.session_state.h_values[cur], step=0.1, key=f"h_{cur}")
E = st.number_input(f"E{to_subscript(cur+1)}", value=st.session_state.E_values[cur], step=0.1, key=f"E_{cur}")

# Само Ed е променлив за всеки пласт
Ed = st.number_input(f"Ed за пласт {cur+1}", value=st.session_state.Ed_values[cur], step=0.1, key=f"Ed_{cur}")

# Запазваме стойностите обратно
st.session_state.h_values[cur] = h
st.session_state.E_values[cur] = E
st.session_state.Ed_values[cur] = Ed

Ed_r = round(Ed, 3)

# Изчисленията се правят само ако натиснем бутон
if st.button("Покажи графиката"):
    h_array = np.array(st.session_state.h_values)
    E_array = np.array(st.session_state.E_values)

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

    # Интерполация и останалата част от графиката (както в оригинала)

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

        st.markdown(f"### Резултат от интерполацията:")
        st.write(f"При Esr/Ei = {target_sr_Ei} и Hn/D = {ratio_r} получаваме y = {round(y_at_ratio, 3)}")

    fig.update_layout(
        title="Графика",
        xaxis_title="H/D",
        yaxis_title="y",
        legend_title="Легенда",
        width=800,
        height=600,
        font=dict(size=14)
    )

    st.plotly_chart(fig)
