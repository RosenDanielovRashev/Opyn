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

# Вече избрано D от падащо меню
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

# Формула за H_{n-1}
st.latex(r"H_{n-1} = \sum_{i=1}^{n-1} h_i")
h_terms = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n-1)])
st.latex(r"H_{n-1} = " + h_terms)
st.write(f"H{to_subscript(n-1)} = {H_n_1:.3f}")

# Формула за H_n
st.latex(r"H_n = \sum_{i=1}^n h_i")
h_terms_n = " + ".join([f"h_{to_subscript(i+1)}" for i in range(n)])
st.latex(r"H_n = " + h_terms_n)
st.write(f"H{to_subscript(n)} = {H_n:.3f}")

# Формула и изчисления за Esr
st.latex(r"Esr = \frac{\sum_{i=1}^{n-1} (E_i \cdot h_i)}{\sum_{i=1}^{n-1} h_i}")

numerator = " + ".join([f"{E_values[i]} \cdot {h_values[i]}" for i in range(n-1)])
denominator = " + ".join([f"{h_values[i]}" for i in range(n-1)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{weighted_sum_n_1}}}{{{sum_h_n_1}}} = {Esr:.3f}"
st.latex(formula_with_values)

# Изчисляване на съотношението H_n / D
ratio = H_n / D if D != 0 else 0
st.latex(r"\frac{H_n}{D} = \frac{" + f"{H_n:.3f}" + "}{" + f"{D:.3f}" + "} = " + f"{ratio:.3f}" )

# Нов параметър Ed (въвеждане без падащо меню, начална стойност 1000)
Ed = st.number_input("Ed", value=1000.0, step=0.1)

# Последен пласт E_n
En = E_values[-1]

# Показване с индекс равен на броя на пластовете (например E₅ ако n=5)
st.markdown("### Изчисления с последен пласт")

st.latex(r"E_{" + str(n) + r"} = " + f"{En:.3f}")

# Изчисления за Esr / En
Esr_over_En = Esr / En if En != 0 else 0
st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr:.3f}" + "}{" + f"{En:.3f}" + "} = " + f"{Esr_over_En:.3f}")

# Изчисления за En / Ed
En_over_Ed = En / Ed if Ed != 0 else 0
st.latex(r"\frac{E_{" + str(n) + r"}}{E_d} = \frac{" + f"{En:.3f}" + "}{" + f"{Ed:.3f}" + "} = " + f"{En_over_Ed:.3f}")

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
            name=f'Ei/Ed = {level}',
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
            name=f'Esr/Ei = {sr_Ei}',
            line=dict(width=2)
        ))

# Търсим къде да поставим точката (target_Hn_D, target_sr_Ei)

target_Hn_D = ratio  # H_n / D
target_sr_Ei = Esr_over_En  # Esr / En

# Намиране между кои изолинии на sr_Ei стойността попада
sr_values_sorted = sorted(unique_sr_Ei)
lower_index = None
for i in range(len(sr_values_sorted)-1):
    if sr_values_sorted[i] <= target_sr_Ei <= sr_values_sorted[i+1]:
        lower_index = i
        break

if lower_index is not None:
    lower_sr = sr_values_sorted[lower_index]
    upper_sr = sr_values_sorted[lower_index + 1]

    # Вземаме изолиниите за долната и горната стойност
    df_lower = df_new[df_new['sr_Ei'] == lower_sr].sort_values(by='H/D')
    df_upper = df_new[df_new['sr_Ei'] == upper_sr].sort_values(by='H/D')

    # Функция за намиране на y при дадено x в изолиния чрез линейна интерполация
    def interp_y(df, x):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        if x <= x_arr[0]:
            return y_arr[0]
        if x >= x_arr[-1]:
            return y_arr[-1]
        for j in range(len(x_arr)-1):
            if x_arr[j] <= x <= x_arr[j+1]:
                # линейна интерполация
                y_interp = y_arr[j] + (y_arr[j+1]-y_arr[j]) * (x - x_arr[j]) / (x_arr[j+1] - x_arr[j])
                return y_interp
        return np.nan

    y_lower = interp_y(df_lower, target_Hn_D)
    y_upper = interp_y(df_upper, target_Hn_D)

    # Изчисляваме средния наклон между точките около x0 (target_Hn_D)
    # Приближение: използваме две точки около x0 от долната изолиния
    def slope_at_x(df, x):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        for j in range(len(x_arr)-1):
            if x_arr[j] <= x <= x_arr[j+1]:
                return (y_arr[j+1] - y_arr[j]) / (x_arr[j+1] - x_arr[j])
        # ако извън интервала
        return 0

    m_lower = slope_at_x(df_lower, target_Hn_D)
    m_upper = slope_at_x(df_upper, target_Hn_D)
    m_avg = (m_lower + m_upper) / 2

    # Линейна интерполация по sr_Ei (параметър t)
    t = (target_sr_Ei - lower_sr) / (upper_sr - lower_sr)

    # Перпендикулярен вектор към изолиниите
    if m_avg != 0:
        m_perp = -1 / m_avg
        perp_vec = np.array([1, m_perp])
        perp_vec = perp_vec / np.linalg.norm(perp_vec)
    else:
        # перпендикулярна линия е вертикална
        perp_vec = np.array([0,1])

    # Център между изолиниите на даденото x
    center = np.array([target_Hn_D, (y_lower + y_upper)/2])

    # Разстоянието между двете изолини по y
    dist = y_upper - y_lower

    # Смещение по перпендикулярната линия спрямо центъра,
    # за да се намери точката на интерполация (t=0.5 е център)
    shift = (t - 0.5) * dist

    # Интерполираната точка
    point = center + shift * perp_vec

    # Добавяне на точката в графиката (точка, не звезда)
    fig.add_trace(go.Scatter(
        x=[point[0]],
        y=[point[1]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Интерполирана точка'
    ))

    # Добавяне на пунктирана линия за интерполация (перпендикулярна към изолиниите)
    line_length = dist * 1.5
    line_start = center - (line_length/2) * perp_vec
    line_end = center + (line_length/2) * perp_vec
    fig.add_trace(go.Scatter(
        x=[line_start[0], line_end[0]],
        y=[line_start[1], line_end[1]],
        mode='lines',
        line=dict(color='red', dash='dot'),
        name='Линия на интерполация'
    ))

    # Добавяне на линия от точката перпендикулярно към абсцисата (H/D)
    fig.add_trace(go.Scatter(
        x=[point[0], point[0]],
        y=[point[1], 0],
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Перпендикуляр към H/D'
    ))
else:
    st.warning("Стойността Esr/En е извън диапазона на изолиниите за интерполация.")

fig.update_layout(
    xaxis_title="H / D",
    yaxis_title="y",
    legend_title="Изолинии",
    width=900,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

