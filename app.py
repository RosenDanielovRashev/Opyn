import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr и H_n/D")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[34.0, 32.04], index=0)

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

sum_h_n_1 = h_array[:-1].sum()
weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0

H_n = h_array.sum()
H_n_1 = sum_h_n_1

st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n-1)])
st.latex(r"H_{n-1} = " + h_terms)
st.write(f"H{to_subscript(n-1)} = {H_n_1:.3f}")

st.latex(r"H_n = \sum_{i=1}^n h_i")
h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n)])
st.latex(r"H_n = " + h_terms_n)
st.write(f"H{to_subscript(n)} = {H_n:.3f}")

st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")

numerator = " + ".join([f"{E_values[i]} · {h_values[i]}" for i in range(n-1)])
denominator = " + ".join([f"{h_values[i]}" for i in range(n-1)])
st.latex(rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum_n_1}}}{{{sum_h_n_1}}} = {Esr:.3f}")

ratio = H_n / D if D != 0 else 0
st.latex(r"\frac{H_n}{D} = \frac{" + f"{H_n:.3f}" + "}{" + f"{D:.3f}" + "} = " + f"{ratio:.3f}")

Ed = st.number_input("Ed", value=1000.0, step=0.1)
En = E_values[-1]

st.markdown("### Изчисления с последен пласт")
st.latex(r"E_{" + str(n) + r"} = " + f"{En:.3f}")

Esr_over_En = Esr / En if En != 0 else 0
st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr:.3f}" + "}{" + f"{En:.3f}" + "} = " + f"{Esr_over_En:.3f}")

En_over_Ed = En / Ed if Ed != 0 else 0
st.latex(r"\frac{E_{" + str(n) + r"}}{E_d} = \frac{" + f"{En:.3f}" + "}{" + f"{Ed:.3f}" + "} = " + f"{En_over_Ed:.3f}")

df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("\u041e\u0440\u0430\u0437\u043c\u0435\u0440\u044f\u0432\u0430\u043d\u0435 \u043d\u0430 \u043e\u043f\u044a\u043d \u0437\u0430 \u043c\u0435\u0436\u0434\u0438\u043d\u0435\u043d \u043f\u043b\u0441\u0442H_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

fig = go.Figure()

if 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'], y=df_level['y'], mode='lines', name=f'Ei/Ed = {level}', line=dict(width=2)
        ))

if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'], y=df_level['y'], mode='lines', name=f'Esr/Ei = {sr_Ei}', line=dict(width=2)
        ))

target_sr_Ei = Esr_over_En
target_Hn_D = ratio

sr_values_sorted = sorted(df_new['sr_Ei'].unique())
lower_index = None
for i in range(len(sr_values_sorted)-1):
    if sr_values_sorted[i] <= target_sr_Ei <= sr_values_sorted[i+1]:
        lower_index = i
        break

if lower_index is not None:
    lower_sr = sr_values_sorted[lower_index]
    upper_sr = sr_values_sorted[lower_index + 1]

    df_lower = df_new[df_new['sr_Ei'] == lower_sr].sort_values(by='H/D')
    df_upper = df_new[df_new['sr_Ei'] == upper_sr].sort_values(by='H/D')

    def interp_xy_perpendicular(df, x0):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        for j in range(len(x_arr)-1):
            if x_arr[j] <= x0 <= x_arr[j+1]:
                t = (x0 - x_arr[j]) / (x_arr[j+1] - x_arr[j])
                y_interp = y_arr[j] + t * (y_arr[j+1] - y_arr[j])
                return np.array([x0, y_interp])
        return np.array([x0, y_arr[-1]])

    point_lower = interp_xy_perpendicular(df_lower, target_Hn_D)
    point_upper = interp_xy_perpendicular(df_upper, target_Hn_D)

    vec = point_upper - point_lower
    t = (target_sr_Ei - lower_sr) / (upper_sr - lower_sr)
    interp_point = point_lower + t * vec

    fig.add_trace(go.Scatter(x=[interp_point[0]], y=[interp_point[1]], mode='markers', marker=dict(color='red', size=10), name='Интерполирана точка'))
    fig.add_trace(go.Scatter(x=[interp_point[0], interp_point[0]], y=[interp_point[1], 0], mode='lines', line=dict(color='blue', dash='dash'), name='Вертикална линия към абсцисата'))

    def interp_x_for_y(df, y_target):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        for k in range(len(y_arr) - 1):
            if (y_arr[k] - y_target) * (y_arr[k+1] - y_target) <= 0:
                t_local = (y_target - y_arr[k]) / (y_arr[k+1] - y_arr[k])
                return x_arr[k] + t_local * (x_arr[k+1] - x_arr[k])
        return None

    Ei_Ed_target = En_over_Ed
    Ei_Ed_values_sorted = sorted(df_original['Ei/Ed'].unique())
    lower_index_EiEd = None
    for i in range(len(Ei_Ed_values_sorted)-1):
        if Ei_Ed_values_sorted[i] <= Ei_Ed_target <= Ei_Ed_values_sorted[i+1]:
            lower_index_EiEd = i
            break

    if lower_index_EiEd is not None:
        lower_level = Ei_Ed_values_sorted[lower_index_EiEd]
        upper_level = Ei_Ed_values_sorted[lower_index_EiEd + 1]
        df_lower_EiEd = df_original[df_original['Ei/Ed'] == lower_level].sort_values(by='H/D')
        df_upper_EiEd = df_original[df_original['Ei/Ed'] == upper_level].sort_values(by='H/D')
        x_lower = interp_x_for_y(df_lower_EiEd, interp_point[1])
        x_upper = interp_x_for_y(df_upper_EiEd, interp_point[1])
        if x_lower is not None and x_upper is not None:
            t_EiEd = (Ei_Ed_target - lower_level) / (upper_level - lower_level)
            x_interp_EiEd = x_lower + t_EiEd * (x_upper - x_lower)
            fig.add_trace(go.Scatter(x=[interp_point[0], x_interp_EiEd], y=[interp_point[1], interp_point[1]], mode='lines', line=dict(color='green', dash='dash'), name='Хоризонтална линия до пресечна точка'))
            fig.add_trace(go.Scatter(x=[x_interp_EiEd], y=[interp_point[1]], mode='markers', marker=dict(color='orange', size=10), name='Пресечна точка с Ei/Ed'))

            horizontal_distance = abs(x_interp_EiEd - 0)
            st.write(f"Разстояние от началото на втората ос до вертикалната линия: {horizontal_distance:.3f}")
        else:
            st.warning("Не може да се намери пресечна точка на хоризонталната линия с изолинията Ei/Ed.")
    else:
        st.warning("Извън интервала на наличните изолинии Ei/Ed за пресичане.")
else:
    st.warning("Esr/Ei не попада между наличните стойности на изолинии.")

fig.update_layout(
    xaxis_title='H/D',
    yaxis_title='y',
    title='Изолинии с интерполации',
    showlegend=False,
    xaxis2=dict(
        overlaying='x',
        side='top',
        range=[0, 1],
        showgrid=False,
        zeroline=False,
        tickmode='auto',
        ticks='outside'
    )
)

st.plotly_chart(fig, use_container_width=True)
