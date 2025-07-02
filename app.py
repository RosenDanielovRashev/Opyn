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

Ed = st.number_input("Ed", value=1000.0, step=0.1)

En = E_values[-1]

ratio = H_n / D if D != 0 else 0
Esr_over_En = Esr / En if En != 0 else 0

# Четене на CSV
df_original = pd.read_csv("danni.csv")
df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)

fig = go.Figure()

# Основни изолинии (първа ос Y)
if 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {level}',
            line=dict(width=2),
            yaxis='y1'  # Основна ос Y
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
            line=dict(width=2, dash='dot'),
            yaxis='y1'
        ))

# Търсене на интерполирана точка (както преди)
target_sr_Ei = Esr_over_En
target_Hn_D = ratio

sr_values_sorted = sorted(df_new['sr_Ei'].unique())
lower_index = None

for i in range(len(sr_values_sorted) - 1):
    if sr_values_sorted[i] <= target_sr_Ei <= sr_values_sorted[i + 1]:
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
        for j in range(len(x_arr) - 1):
            if x_arr[j] <= x0 <= x_arr[j + 1]:
                p1 = np.array([x_arr[j], y_arr[j]])
                p2 = np.array([x_arr[j + 1], y_arr[j + 1]])
                seg_vec = p2 - p1
                seg_len = np.linalg.norm(seg_vec)
                if seg_len == 0:
                    return p1
                t = (x0 - x_arr[j]) / (x_arr[j + 1] - x_arr[j])
                point_on_seg = p1 + t * seg_vec
                return point_on_seg
        # Ако x0 извън диапазона:
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

    # Добавяне на червена точка на първа ос Y
    fig.add_trace(go.Scatter(
        x=[interp_point[0]],
        y=[interp_point[1]],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Интерполирана точка',
        yaxis='y1'
    ))

    # Оранжева вертикална линия от оранжевата точка (interp_point) до y=2.5
    fig.add_trace(go.Scatter(
        x=[interp_point[0], interp_point[0]],
        y=[interp_point[1], 2.5],
        mode='lines',
        line=dict(color='orange', width=3, dash='dot'),
        showlegend=False,
        yaxis='y1'
    ))

# Добавяне на прозрачно трасе на втора ос Y
# За пример ще добавя синя точка/линия със стойности от df_new със същия x
df_transparent = df_new[df_new['sr_Ei'] == lower_sr] if lower_index is not None else df_new.iloc[:10]
fig.add_trace(go.Scatter(
    x=df_transparent['H/D'],
    y=df_transparent['y']*0.8,  # например някакво мащабиране за визуализация
    mode='lines',
    line=dict(color='blue', width=2),
    opacity=0.3,
    name='Прозрачно трасе',
    yaxis='y2'
))

# Оформление на две оси Y
fig.update_layout(
    yaxis=dict(
        title="Y (ос 1)",
        range=[0, 3],
        side='left',
        showgrid=True,
    ),
    yaxis2=dict(
        title="Втора ос Y",
        overlaying='y',
        side='right',
        showgrid=False,
        range=[0, 3],
    ),
    xaxis=dict(
        title="H/D",
        range=[0, max(df_new['H/D'].max(), ratio)*1.1]
    ),
    showlegend=False,  # махаме легендата
    height=600,
    margin=dict(t=50, b=100)
)

# Показване на sigma r под графиката
st.plotly_chart(fig, use_container_width=True)
st.markdown(f"<div style='text-align:center; font-size:18px; margin-top:10px;'>σᵣ = {interp_point[0]}</div>", unsafe_allow_html=True)
