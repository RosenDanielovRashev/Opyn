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

# Изчисляване на Esr за първите n-1 пласта
sum_h_n_1 = h_array[:-1].sum()
weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 != 0 else 0

# Изчисляване на H_n и H_{n-1}
H_n = h_array.sum()
H_n_1 = sum_h_n_1

# Закръгляне
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

numerator = " + ".join([f"{round(E_values[i],3)} \cdot {round(h_values[i],3)}" for i in range(n-1)])
denominator = " + ".join([f"{round(h_values[i],3)}" for i in range(n-1)])
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{round(weighted_sum_n_1,3)}}}{{{round(sum_h_n_1,3)}}} = {Esr_r}"
st.latex(formula_with_values)

ratio_display = rf"\frac{{H_n}}{{D}} = \frac{{{H_n_r}}}{{{round(D,3)}}} = {ratio_r}"
st.latex(ratio_display)

Ed = st.number_input("Ed", value=100.0, step=0.1)
Ed_r = round(Ed, 3)

En = E_values[-1]
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

# --- Търсене на интервал между две изолинии за Esr/Ei (Esr_over_En)
target_sr_Ei = Esr_over_En_r
target_Hn_D = ratio_r  # Hn/D

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
                p1 = np.array([x_arr[j], y_arr[j]])
                p2 = np.array([x_arr[j+1], y_arr[j+1]])
                seg_vec = p2 - p1
                seg_len = np.linalg.norm(seg_vec)
                if seg_len == 0:
                    return p1
                t = (x0 - x_arr[j]) / (x_arr[j+1] - x_arr[j])
                point_on_seg = p1 + t * seg_vec
                return point_on_seg
        if x0 < x_arr[0]:
            return np.array([x_arr[0], y_arr[0]])
        else:
            return np.array([x_arr[-1], y_arr[-1]])

    point_lower = interp_xy_perpendicular(df_lower, target_Hn_D)
    point_upper = interp_xy_perpendicular(df_upper, target_Hn_D)

    vec = point_upper - point_lower

    t = (target_sr_Ei - lower_sr) / (upper_sr - lower_sr)

    interp_point = point_lower + t * vec

    # **Промяна на образуването на червената точка**:
    # x е зададено на ratio_r (Hn/D)
    red_x = ratio_r

    # y се изчислява чрез интерполация по vertical line през Hn/D между двете изолинии (interp_point[1] вече е стойност на y)
    red_y = interp_point[1]

    # Добавяне на червената точка с новите координати (x = Hn/D, y = интерполирана стойност)
    fig.add_trace(go.Scatter(
        x=[red_x],
        y=[round(red_y,3)],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Интерполирана червена точка'
    ))

    # Добавяне на вертикална линия от червената точка до x-оста
    fig.add_trace(go.Scatter(
        x=[red_x, red_x],
        y=[round(red_y,3), 0],
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Вертикална линия към абсцисата'
    ))

    # Функция за обратна интерполация - намира x за дадено y по изолинията
    def interp_x_for_y(df, y_target):
        x_arr = df['H/D'].values
        y_arr = df['y'].values
        for k in range(len(y_arr) - 1):
            y1, y2 = y_arr[k], y_arr[k + 1]
            if (y1 - y_target) * (y2 - y_target) <= 0:  # y_target между y1 и y2
                x1, x2 = x_arr[k], x_arr[k + 1]
                if y2 == y1:
                    return x1
                t_local = (y_target - y1) / (y2 - y1)
                x_interp = x1 + t_local * (x2 - x1)
                return x_interp
        return None

    # Намиране на изолиниите в df_original за най-близки нива на Ei/Ed
    Ei_Ed_target = En_over_Ed_r
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

        x_lower = interp_x_for_y(df_lower_EiEd, round(red_y,3))
        x_upper = interp_x_for_y(df_upper_EiEd, round(red_y,3))

        if x_lower is not None and x_upper is not None:
            t_EiEd = (Ei_Ed_target - lower_level) / (upper_level - lower_level)
            x_interp_EiEd = x_lower + t_EiEd * (x_upper - x_lower)

            # Показване на стойности в страничната лента
            st.sidebar.markdown(f"## Информация за червената точка")
            st.sidebar.write(f"Esr/Ei = {target_sr_Ei}")
            st.sidebar.write(f"Hn/D = {ratio_r}")
            st.sidebar.write(f"Интерполирана y (червена точка) = {round(red_y,3)}")
            st.sidebar.write(f"Интерполирано x по Ei/Ed = {round(x_interp_EiEd,3)}")

# Добавяне на оранжева точка - Hn/D, En/Ed
fig.add_trace(go.Scatter(
    x=[ratio_r],
    y=[En_over_Ed_r],
    mode='markers',
    marker=dict(color='orange', size=10),
    name='Точка Hn/D и En/Ed'
))

fig.update_layout(
    title="Комбинирани изолинии",
    xaxis_title="Hn/D",
    yaxis_title="Стойности",
    legend_title="Изолинии",
    font=dict(size=14)
)

st.plotly_chart(fig, use_container_width=True)
