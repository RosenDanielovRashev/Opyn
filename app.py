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

# Въвеждане на h_i и E_i за всеки пласт
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
Esr_r = round(Esr, 3)
H_n_r = round(H_n, 3)
ratio = H_n / D if D != 0 else 0
ratio_r = round(ratio, 3)

# Формули и стойности
st.latex(r"H_n = \sum h_i \Rightarrow H_n = " + f"{H_n_r}")
st.latex(r"Esr = \frac{\sum E_i h_i}{\sum h_i} \Rightarrow Esr = " + f"{Esr_r}")
st.latex(r"\frac{H_n}{D} = " + f"\frac{{{H_n_r}}}{{{D}}} = {ratio_r}")

Ed = st.number_input("Ed", value=100.0, step=0.1)
Ed_r = round(Ed, 3)
En = E_values[-1]
En_r = round(En, 3)

Esr_over_En = Esr / En if En != 0 else 0
Esr_over_En_r = round(Esr_over_En, 3)
En_over_Ed = En / Ed if Ed != 0 else 0
En_over_Ed_r = round(En_over_Ed, 3)

st.latex(r"\frac{Esr}{E_n} = " + f"{Esr_r}/{En_r} = {Esr_over_En_r}")
st.latex(r"\frac{E_n}{E_d} = " + f"{En_r}/{Ed_r} = {En_over_Ed_r}")

# Зареждане на CSV файлове
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

fig = go.Figure()

# Графика - оригинални изолинии Ei/Ed
if 'Ei/Ed' in df_original.columns:
    for val in sorted(df_original['Ei/Ed'].unique()):
        df_val = df_original[df_original['Ei/Ed'] == val].sort_values(by='H/D')
        fig.add_trace(go.Scatter(x=df_val['H/D'], y=df_val['y'], mode='lines',
                                 name=f"Ei/Ed = {round(val,3)}", line=dict(width=2)))

# Графика - изолинии Esr/Ei
if 'sr_Ei' in df_new.columns:
    for val in sorted(df_new['sr_Ei'].unique()):
        df_val = df_new[df_new['sr_Ei'] == val].sort_values(by='H/D')
        fig.add_trace(go.Scatter(x=df_val['H/D'], y=df_val['y'], mode='lines',
                                 name=f"Esr/Ei = {round(val,3)}", line=dict(width=2)))

# === Новата логика: вертикална линия и червена точка ===
target_sr_Ei = Esr_over_En_r
target_Hn_D = ratio_r

df_target = df_new[df_new['sr_Ei'] == target_sr_Ei].sort_values(by='H/D')

if df_target.empty:
    st.error(f"❌ Няма изолиния със стойност Esr/Ei = {target_sr_Ei}.")
else:
    x_vals = df_target['H/D'].values
    y_vals = df_target['y'].values

    epsilon = 1e-6
    x_min, x_max = np.min(x_vals), np.max(x_vals)

    if target_Hn_D < x_min - epsilon or target_Hn_D > x_max + epsilon:
        st.error(f"❌ Hn/D = {target_Hn_D} е извън обхвата на изолинията Esr/Ei = {target_sr_Ei}.")
    else:
        def interpolate_y(x, x_arr, y_arr):
            for i in range(len(x_arr)-1):
                if x_arr[i] - epsilon <= x <= x_arr[i+1] + epsilon:
                    t = (x - x_arr[i]) / (x_arr[i+1] - x_arr[i])
                    return y_arr[i] + t * (y_arr[i+1] - y_arr[i])
            return None

        y_at_ratio = interpolate_y(target_Hn_D, x_vals, y_vals)

        if y_at_ratio is None:
            st.error("❌ Неуспешна интерполация при Hn/D.")
        else:
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
                name='Червена точка'
            ))

            # Продължение: намиране на пресечка с Ei/Ed
            Ei_Ed_target = En_over_Ed_r
            Ei_Ed_vals = sorted(df_original['Ei/Ed'].unique())
            lvl1, lvl2 = None, None

            for i in range(len(Ei_Ed_vals)-1):
                if Ei_Ed_vals[i] <= Ei_Ed_target <= Ei_Ed_vals[i+1]:
                    lvl1 = Ei_Ed_vals[i]
                    lvl2 = Ei_Ed_vals[i+1]
                    break

            if lvl1 and lvl2:
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
                        name='σr линия'
                    ))

                    sigma_r = x_interp / 2
                    st.markdown(f"**σr = {round(sigma_r, 3)}**")
                else:
                    st.warning("⚠️ Не можа да се намери x за даден y върху изолиниите Ei/Ed.")
            else:
                st.warning("⚠️ Стойността Ei/Ed е извън наличните интервали.")
