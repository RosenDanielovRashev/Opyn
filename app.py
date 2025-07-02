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
Esr = round(weighted_sum_n_1 / sum_h_n_1, 3) if sum_h_n_1 != 0 else 0

# Изчисляване на H_n и H_{n-1}
H_n = round(h_array.sum(), 3)
H_n_1 = round(sum_h_n_1, 3)

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
formula_with_values = rf"Esr = \frac{{{numerator}}}{{{denominator}}} = \frac{{{round(weighted_sum_n_1,3)}}}{{{round(sum_h_n_1,3)}}} = {Esr:.3f}"
st.latex(formula_with_values)

ratio = round(H_n / D, 3) if D != 0 else 0
st.latex(r"\frac{H_n}{D} = \frac{" + f"{H_n:.3f}" + "}{" + f"{D:.3f}" + "} = " + f"{ratio:.3f}" )

Ed = st.number_input("Ed", value=1000.0, step=0.1)

En = E_values[-1]

st.markdown("### Изчисления с последен пласт")

st.latex(r"E_{" + str(n) + r"} = " + f"{En:.3f}")

Esr_over_En = round(Esr / En, 3) if En != 0 else 0
st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr:.3f}" + "}{" + f"{En:.3f}" + "} = " + f"{Esr_over_En:.3f}")

En_over_Ed = round(En / Ed, 3) if Ed != 0 else 0
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

# Търсене на интервала за Esr_over_En (т.е. target_sr_Ei)
target_sr_Ei = Esr_over_En
target_Hn_D = ratio  # Hn/D

sr_values_sorted = sorted(df_new['sr_Ei'].unique())
lower_index = None

# По-точно търсене на интервала между две стойности sr_Ei
for i in range(len(sr_values_sorted) - 1):
    low = sr_values_sorted[i]
    high = sr_values_sorted[i + 1]
    if low <= target_sr_Ei < high:
        lower_index = i
        break
# Ако стойността е равна на най-голямата, вземаме последния интервал
if lower_index is None and target_sr_Ei == sr_values_sorted[-1]:
    lower_index = len(sr_values_sorted) - 2

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
                return np.round(point_on_seg, 3)
        # Ако x0 извън обхвата, вземаме краен край
        if x0 < x_arr[0]:
            return np.array([x_arr[0], y_arr[0]])
        else:
            return np.array([x_arr[-1], y_arr[-1]])

    point_lower = interp_xy_perpendicular(df_lower, target_Hn_D)
    point_upper = interp_xy_perpendicular(df_upper, target_Hn_D)

    vec = point_upper - point_lower

    t = round((target_sr_Ei - lower_sr) / (upper_sr - lower_sr), 3)

    interp_point = point_lower + t * vec
    interp_point = np.round(interp_point, 3)

    # Добавяне на интерполирана точка (първа точка)
    fig.add_trace(go.Scatter(
        x=[interp_point[0]],
        y=[interp_point[1]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Интерполирана точка'
    ))

    # Вертикална линия към абсцисата (x-оста)
    fig.add_trace(go.Scatter(
        x=[interp_point[0], interp_point[0]],
        y=[interp_point[1], 0],
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Вертикална линия към x-оста'
    ))

    # Хоризонтална линия от интерполираната точка до y=0 на x=0
    fig.add_trace(go.Scatter(
        x=[0, interp_point[0]],
        y=[interp_point[1], interp_point[1]],
        mode='lines',
        line=dict(color='green', dash='dot'),
        name='Хоризонтална линия'
    ))

    # Търсим пресечна точка на хоризонталната линия с изолиния, равна на Ei/Ed

    # Първо намираме изолинията за Ei/Ed в df_original
    Ei_Ed_target = Esr / Ed  # може да си го настроиш ако трябва

    Ei_Ed_sorted = sorted(df_original['Ei/Ed'].unique())
    lower_index_EiEd = None
    for i in range(len(Ei_Ed_sorted) - 1):
        low = Ei_Ed_sorted[i]
        high = Ei_Ed_sorted[i + 1]
        if low <= Ei_Ed_target < high:
            lower_index_EiEd = i
            break
    if lower_index_EiEd is None and Ei_Ed_target == Ei_Ed_sorted[-1]:
        lower_index_EiEd = len(Ei_Ed_sorted) - 2

    if lower_index_EiEd is not None:
        lower_level = Ei_Ed_sorted[lower_index_EiEd]
        upper_level = Ei_Ed_sorted[lower_index_EiEd + 1]

        df_lower_EiEd = df_original[df_original['Ei/Ed'] == lower_level].sort_values(by='H/D')
        df_upper_EiEd = df_original[df_original['Ei/Ed'] == upper_level].sort_values(by='H/D')

        def interp_y_at_H_D(df, y_target):
            # Интерполира x за дадено y (обратна интерполация)
            x_arr = df['H/D'].values
            y_arr = df['y'].values
            for k in range(len(y_arr)-1):
                if y_arr[k] >= y_target >= y_arr[k+1] or y_arr[k] <= y_target <= y_arr[k+1]:
                    p1 = np.array([x_arr[k], y_arr[k]])
                    p2 = np.array([x_arr[k+1], y_arr[k+1]])
                    # Линейна интерполация за x
                    if (y_arr[k+1] - y_arr[k]) == 0:
                        return p1[0]
                    t_local = (y_target - y_arr[k]) / (y_arr[k+1] - y_arr[k])
                    x_interp = p1[0] + t_local * (p2[0] - p1[0])
                    return round(x_interp, 3)
            # Ако няма пресечна точка в интервала
            return None

        x_lower = interp_y_at_H_D(df_lower_EiEd, interp_point[1])
        x_upper = interp_y_at_H_D(df_upper_EiEd, interp_point[1])

        if x_lower is not None and x_upper is not None:
            t_EiEd = round((Ei_Ed_target - lower_level) / (upper_level - lower_level), 3)
            x_interp_EiEd = round(x_lower + t_EiEd * (x_upper - x_lower), 3)

            # Добавяне на точка на пресечната с изолинията Ei/Ed
            fig.add_trace(go.Scatter(
                x=[x_interp_EiEd],
                y=[interp_point[1]],
                mode='markers',
                marker=dict(color='orange', size=10),
                name='Пресечна точка с Ei/Ed'
            ))

st.plotly_chart(fig, use_container_width=True)
