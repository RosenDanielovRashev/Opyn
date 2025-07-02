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

# --- Търсене на интервал между две изолинии за Esr/Ei (Esr_over_En)
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
    interp_point[0] = round(interp_point[0], 3)
    interp_point[1] = round(interp_point[1], 3)

    # Добавяне на интерполирана точка (първа точка)
    fig.add_trace(go.Scatter(
        x=[interp_point[0]],
        y=[interp_point[1]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Интерполирана точка'
    ))

    # Добавяне на вертикална линия от точката до x-оста (x=interp_point[0], y от interp_point[1] до 0)
    fig.add_trace(go.Scatter(
        x=[interp_point[0], interp_point[0]],
        y=[interp_point[1], 0],
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
                    return round(x1, 3)
                t_local = (y_target - y1) / (y2 - y1)
                x_interp = x1 + t_local * (x2 - x1)
                return round(x_interp, 3)
        return None

    # Намиране на изолиниите в df_original за най-близки нива на Ei/Ed
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
            x_interp_EiEd = round(x_lower + t_EiEd * (x_upper - x_lower), 3)

            # Добавяне на хоризонтална линия от първата точка до y=interp_point[1]
            fig.add_trace(go.Scatter(
                x=[interp_point[0], x_interp_EiEd],
                y=[interp_point[1], interp_point[1]],
                mode='lines',
                line=dict(color='green', dash='dash'),
                name='Хоризонтална линия до пресечна точка'
            ))

            # Добавяне на оранжева точка на пресичане хоризонтална линия с изолиния Ei/Ed
            fig.add_trace(go.Scatter(
                x=[x_interp_EiEd],
                y=[interp_point[1]],
                mode='markers',
                marker=dict(color='orange', size=10),
                name='Пресечна точка с Ei/Ed'
            ))

            # Добавяне на вертикална линия от оранжевата точка до y=2.5 (твоето искане)
            fig.add_trace(go.Scatter(
                x=[x_interp_EiEd, x_interp_EiEd],
                y=[interp_point[1], 2.5],
                mode='lines',
                line=dict(color='orange', dash='dot'),
                name='Вертикална линия от оранжева точка до y=2.5'
            ))

# --- Добавяне на невидим trace за втората ос (за да се покаже мащабът)
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[None, None],  # y не влияе
    mode='lines',
    line=dict(color='rgba(0,0,0,0)'),
    showlegend=False,
    hoverinfo='skip',
    xaxis='x2'  # Свързваме с втората ос
))

fig.update_layout(
    title='Графика на изолинии',
    xaxis=dict(
        title='H/D',
        showgrid=True,
        zeroline=False,
    ),
    xaxis2=dict(
        overlaying='x',
        side='top',
        range=[fig.layout.xaxis.range[0] if fig.layout.xaxis.range else None, 1],
        showgrid=False,
        zeroline=False,
        tickvals=[0, 0.25, 0.5, 0.75, 1],
        ticktext=['0', '0.25', '0.5', '0.75', '1']
    ),
    yaxis=dict(
        title='y',
    ),
    showlegend=False
)

st.plotly_chart(fig)

# Проверка дали x_interp_EiEd е дефинирана и не е None
if ('x_interp_EiEd' in locals()) and (x_interp_EiEd is not None):
    sigma_r = x_interp_EiEd / 3
    st.markdown(f"**σ_r = {sigma_r:.3f}**")
else:
    st.markdown("**σ_r = -** (Няма изчислена стойност)")
