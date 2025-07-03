import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.title("Комбинирани изолинии с изчисление на Esr и H_n/D")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Входни параметри
n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
D = st.selectbox("Избери D", options=[32.04, 34.0], index=0)

# Въвеждане на h_i и E_i
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

# Изчисления
sum_h_n_1 = h_array[:-1].sum()
weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0

H_n = h_array.sum()
H_n_1 = sum_h_n_1
H_n_r = round(H_n, 3)
H_n_1_r = round(H_n_1, 3)
Esr_r = round(Esr, 3)
ratio = H_n / D if D != 0 else 0
ratio_r = round(ratio, 3)

# Latex изрази
st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
st.write(f"H{to_subscript(n-1)} = {H_n_1_r}")
st.latex(r"H_n = \sum_{i=1}^{n} h_i")
st.write(f"H{to_subscript(n)} = {H_n_r}")

numerator = " + ".join([f"{round(E_values[i],3)} \cdot {round(h_values[i],3)}" for i in range(n-1)])
denominator = " + ".join([f"{round(h_values[i],3)}" for i in range(n-1)])
st.latex(rf"Esr = \frac{{{numerator}}}{{{denominator}}} = {Esr_r}")
st.latex(rf"\frac{{H_n}}{{D}} = \frac{{{H_n_r}}}{{{round(D,3)}}} = {ratio_r}")

Ed = st.number_input("Ed", value=100.0, step=0.1)
Ed_r = round(Ed, 3)

En = E_values[-1]
En_r = round(En, 3)
Esr_over_En = Esr / En if En != 0 else 0
Esr_over_En_r = round(Esr_over_En, 3)
En_over_Ed = En / Ed if Ed != 0 else 0
En_over_Ed_r = round(En_over_Ed, 3)

st.latex(r"E_{" + str(n) + r"} = " + f"{En_r}")
st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr_r}" + "}{" + f"{En_r}" + "} = " + f"{Esr_over_En_r}")
st.latex(r"\frac{E_{" + str(n) + r"}}{E_d} = \frac{" + f"{En_r}" + "}{" + f"{Ed_r}" + "} = " + f"{En_over_Ed_r}")

# Зареждане на данни
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

# Графика
fig = go.Figure()

# Изолинии Ei/Ed
if 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {round(level,3)}'
        ))

# Изолинии Esr/Ei
if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Esr/Ei = {round(sr,3)}'
        ))

# === Начертай вертикална линия при Hn/D и намери пресечна точка с Esr/Ei ===
target_sr_Ei = Esr_over_En_r
target_Hn_D = ratio_r

df_target_isoline = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')

if df_target_isoline.empty:
    st.error(f"Не е намерена изолиния със стойност Esr/Ei = {target_sr_Ei}. Няма да се постави червена точка.")
else:
    x_vals = df_target_isoline['H/D'].values
    y_vals = df_target_isoline['y'].values

    if target_Hn_D < x_vals.min() or target_Hn_D > x_vals.max():
        st.error(f"Hn/D = {target_Hn_D} е извън обхвата на изолинията със стойност Esr/Ei = {target_sr_Ei}.")
    else:
        # Интерполация за y при дадено x
        def interpolate_y(x, x_arr, y_arr):
            for i in range(len(x_arr) - 1):
                if x_arr[i] <= x <= x_arr[i+1]:
                    t = (x - x_arr[i]) / (x_arr[i+1] - x_arr[i])
                    return y_arr[i] + t * (y_arr[i+1] - y_arr[i])
            return None

        y_at_ratio = interpolate_y(target_Hn_D, x_vals, y_vals)
        if y_at_ratio is not None:
            y_at_ratio_r = round(y_at_ratio, 3)

            # Вертикална линия
            fig.add_trace(go.Scatter(
                x=[target_Hn_D, target_Hn_D],
                y=[0, y_at_ratio_r],
                mode='lines',
                line=dict(color='blue', dash='dash'),
                name='Вертикална линия Hn/D'
            ))

            # Червена точка
            fig.add_trace(go.Scatter(
                x=[target_Hn_D],
                y=[y_at_ratio_r],
                mode='markers',
                marker=dict(color='red', size=10),
                name='Червена точка (пресичане Esr/Ei)'
            ))

            # Точка за следваща част
            interp_point = np.array([target_Hn_D, y_at_ratio_r])

            # === Продължение: намиране на пресечна точка с Ei/Ed ===
            Ei_Ed_target = En_over_Ed_r
            Ei_Ed_values_sorted = sorted(df_original['Ei/Ed'].unique())

            lower_index = None
            for i in range(len(Ei_Ed_values_sorted)-1):
                if Ei_Ed_values_sorted[i] <= Ei_Ed_target <= Ei_Ed_values_sorted[i+1]:
                    lower_index = i
                    break

            if lower_index is not None:
                lvl1 = Ei_Ed_values_sorted[lower_index]
                lvl2 = Ei_Ed_values_sorted[lower_index + 1]
                df1 = df_original[df_original['Ei/Ed'] == lvl1].sort_values(by='H/D')
                df2 = df_original[df_original['Ei/Ed'] == lvl2].sort_values(by='H/D')

                def get_x_for_y(df, y_target):
                    x_arr = df['H/D'].values
                    y_arr = df['y'].values
                    for j in range(len(y_arr)-1):
                        if (y_arr[j] - y_target) * (y_arr[j+1] - y_target) <= 0:
                            t = (y_target - y_arr[j]) / (y_arr[j+1] - y_arr[j])
                            return x_arr[j] + t * (x_arr[j+1] - x_arr[j])
                    return None

                x1 = get_x_for_y(df1, y_at_ratio_r)
                x2 = get_x_for_y(df2, y_at_ratio_r)

                if x1 and x2:
                    t = (Ei_Ed_target - lvl1) / (lvl2 - lvl1)
                    x_interp = x1 + t * (x2 - x1)

                    fig.add_trace(go.Scatter(
                        x=[target_Hn_D, x_interp],
                        y=[y_at_ratio_r, y_at_ratio_r],
                        mode='lines',
                        line=dict(color='green', dash='dash'),
                        name='Хоризонтална линия'
                    ))

                    fig.add_trace(go.Scatter(
                        x=[x_interp],
                        y=[y_at_ratio_r],
                        mode='markers',
                        marker=dict(color='orange', size=10),
                        name='Оранжева точка'
                    ))

                    fig.add_trace(go.Scatter(
                        x=[x_interp, x_interp],
                        y=[y_at_ratio_r, 2.5],
                        mode='lines',
                        line=dict(color='orange', dash='dot'),
                        name='Вертикална линия до y=2.5'
                    ))

                    sigma_r = x_interp / 2
                    st.markdown(f"**σr = {round(sigma_r, 3)}**")
                else:
                    st.warning("Не можа да се изчисли пресечна точка с изолиния Ei/Ed.")
        else:
            st.error("Неуспешна интерполация при Hn/D.")

fig.update_layout(
    title="Графика на изолинии",
    xaxis_title="H/D",
    yaxis_title="y",
    showlegend=True
)

st.plotly_chart(fig)
