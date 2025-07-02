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
D = st.selectbox("Избери D", options=[34.0, 32.04], index=0)

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

# Изчисляване на Esr за първите n-1 пласта
sum_h_n_1 = h_array[:-1].sum()
weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0

# Изчисляване на H_n и H_{n-1}
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

numerator = " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(n-1)])
denominator = " + ".join([f"{h_values[i]}" for i in range(n-1)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum_n_1}}}{{{sum_h_n_1}}} = {Esr:.3f}"
st.latex(formula_with_values)

ratio = H_n / D if D != 0 else 0
st.latex(r"\frac{H_n}{D} = \frac{" + f"{H_n:.3f}" + "}{" + f"{D:.3f}" + "} = " + f"{ratio:.3f}" )

Ed = st.number_input("Ed", value=1000.0, step=0.1)
En = E_values[-1]

st.markdown("### Изчисления с последен пласт")

st.latex(r"E_{" + str(n) + r"} = " + f"{En:.3f}")

Esr_over_En = Esr / En if En != 0 else 0
st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr:.3f}" + "}{" + f"{En:.3f}" + "} = " + f"{Esr_over_En:.3f}")

En_over_Ed = En / Ed if Ed != 0 else 0
st.latex(r"\frac{E_{" + str(n) + r"}}{E_d} = \frac{" + f"{En:.3f}" + "}{" + f"{Ed:.3f}" + "} = " + f"{En_over_Ed:.3f}")

# Зареждане на данни и построяване на графика
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

fig = go.Figure()

# Добавяне на изолинии от първия файл (Ei/Ed)
if 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {level}',
            line=dict(width=2)
        ))

# Добавяне на изолинии от втория файл (Esr/Ei)
if 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    for sr_Ei in unique_sr_Ei:
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Esr/Ei = {sr_Ei}',
            line=dict(width=2)
        ))

# --- Интерполация на точката в df_new за Esr_over_En и Hn/D ---
target_sr_Ei = Esr_over_En
target_Hn_D = ratio  # Hn/D

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

    def interp_xy(df, x0):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        for j in range(len(x_arr)-1):
            if x_arr[j] <= x0 <= x_arr[j+1]:
                x1, x2 = x_arr[j], x_arr[j+1]
                y1, y2 = y_arr[j], y_arr[j+1]
                t = (x0 - x1) / (x2 - x1)
                y_interp = y1 + t * (y2 - y1)
                return y_interp
        # Ако x0 извън диапазона, връщаме най-близката стойност
        if x0 < x_arr[0]:
            return y_arr[0]
        else:
            return y_arr[-1]

    y_lower = interp_xy(df_lower, target_Hn_D)
    y_upper = interp_xy(df_upper, target_Hn_D)

    # Интерполация по sr_Ei
    t_sr = (target_sr_Ei - lower_sr) / (upper_sr - lower_sr)
    y_interp = y_lower + t_sr * (y_upper - y_lower)

    interp_point = np.array([target_Hn_D, y_interp])

    # Добавяне на интерполирана точка
    fig.add_trace(go.Scatter(
        x=[interp_point[0]],
        y=[interp_point[1]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Интерполирана точка'
    ))

    # --- Търсим точка в df_original за избран Ei/Ed, където y = 0 (линейна интерполация) ---
    # Избираме най-близкото Ei/Ed
    closest_Ei_Ed = min(unique_Ei_Ed, key=lambda x: abs(x - En_over_Ed))
    df_orig_level = df_original[df_original['Ei/Ed'] == closest_Ei_Ed].sort_values(by='H/D')

    # Функция за линейна интерполация за намиране на x при y=0
    def find_x_at_y_zero(df):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        for k in range(len(y_arr)-1):
            if (y_arr[k] >= 0 >= y_arr[k+1]) or (y_arr[k] <= 0 <= y_arr[k+1]):
                # Линейна интерполация за x при y=0
                x1, x2 = x_arr[k], x_arr[k+1]
                y1, y2 = y_arr[k], y_arr[k+1]
                t = (0 - y1) / (y2 - y1)
                x_zero = x1 + t * (x2 - x1)
                return x_zero
        # Ако няма пресичане с y=0 връщаме None
        return None

    x_at_y_zero = find_x_at_y_zero(df_orig_level)

    if x_at_y_zero is not None:
        # Втора точка на абсцисата
        zero_point = np.array([x_at_y_zero, 0])

        # Добавяне на вертикална линия от interp_point до zero_point
        fig.add_trace(go.Scatter(
            x=[interp_point[0], zero_point[0]],
            y=[interp_point[1], zero_point[1]],
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Линия между интерполираната точка и абсцисата'
        ))

        # Маркираме и втората точка
        fig.add_trace(go.Scatter(
            x=[zero_point[0]],
            y=[zero_point[1]],
            mode='markers',
            marker=dict(color='green', size=10),
            name=f'Точка на y=0 за Ei/Ed={closest_Ei_Ed:.3f}'
        ))
    else:
        st.warning(f"Не е намерено пресичане с y=0 за Ei/Ed={closest_Ei_Ed:.3f} в df_original.")
else:
    st.warning("Стойността Esr/Ei е извън диапазона на изолиниите за интерполация.")

fig.update_layout(
    xaxis_title="H / D",
    yaxis_title="y",
    legend_title="Изолинии",
    width=900,
    height=600,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
