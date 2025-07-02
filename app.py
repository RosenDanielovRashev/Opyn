import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")
st.title("Комбинирани изолинии с изчисление на Esr и Hₙ/D")

def to_subscript(number):
    subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscripts)

# Входни параметри
with st.expander("Входни параметри", expanded=True):
    n = st.number_input("Брой пластове (n)", min_value=2, step=1, value=3)
    D = st.selectbox("Избери D", options=[34.0, 32.04], index=0)
    Ed = st.number_input("Ed", value=1000.0, step=0.1)

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
with st.expander("Изчислителни резултати", expanded=True):
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
    st.latex(r"\frac{H_n}{D} = \frac{" + f"{H_n:.3f}" + "}{" + f"{D:.3f}" + "} = " + f"{ratio:.3f}")

    En = E_values[-1]
    st.latex(r"E_{" + str(n) + r"} = " + f"{En:.3f}")

    Esr_over_En = Esr / En if En != 0 else 0
    st.latex(r"\frac{Esr}{E_{" + str(n) + r"}} = \frac{" + f"{Esr:.3f}" + "}{" + f"{En:.3f}" + "} = " + f"{Esr_over_En:.3f}")

    En_over_Ed = En / Ed if Ed != 0 else 0
    st.latex(r"\frac{E_{" + str(n) + r"}}{E_d} = \frac{" + f"{En:.3f}" + "}{" + f"{Ed:.3f}" + "} = " + f"{En_over_Ed:.3f}")

# Зареждане на данни и построяване на графика
@st.cache_data
def load_data():
    df_original = pd.read_csv("danni.csv")
    df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)
    return df_original, df_new

df_original, df_new = load_data()

fig = go.Figure()

# Добавяне на изолинии от оригиналните данни
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
        )

# Добавяне на изолинии от новите данни
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
        )

# Интерполационна логика
if 'sr_Ei' in df_new.columns:
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

        # Добавяне на интерполирана точка
        fig.add_trace(go.Scatter(
            x=[interp_point[0]],
            y=[interp_point[1]],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Интерполирана точка'
        ))

        # Добавяне на вертикална линия до абсцисата
        fig.add_trace(go.Scatter(
            x=[interp_point[0], interp_point[0]],
            y=[interp_point[1], 0],
            mode='lines',
            line=dict(color='blue', dash='dash'),
            name='Вертикална линия'
        ))

        # Намиране на изолинията Ei/Ed
        if 'Ei/Ed' in df_original.columns:
            # Намиране на най-близката изолиния Ei/Ed
            closest_Ei_Ed = min(unique_Ei_Ed, key=lambda x: abs(x - En_over_Ed))
            df_closest = df_original[df_original['Ei/Ed'] == closest_Ei_Ed].sort_values(by='H/D')
            
            # Намиране на пресечната точка с изолинията Ei/Ed
            x_arr_closest = df_closest['H/D'].values
            y_arr_closest = df_closest['y'].values
            
            # Намиране на y-стойността на изолинията за x = interp_point[0]
            y_target = np.interp(interp_point[0], x_arr_closest, y_arr_closest)
            
            # Добавяне на хоризонтална линия до изолинията Ei/Ed
            fig.add_trace(go.Scatter(
                x=[interp_point[0], interp_point[0]],
                y=[interp_point[1], y_target],
                mode='lines',
                line=dict(color='green', dash='dashdot'),
                name=f'Хоризонтална линия до Ei/Ed={closest_Ei_Ed:.3f}'
            ))
            
            # Добавяне на маркер в пресечната точка
            fig.add_trace(go.Scatter(
                x=[interp_point[0]],
                y=[y_target],
                mode='markers',
                marker=dict(color='purple', size=8),
                name=f'Пресечна точка с Ei/Ed={closest_Ei_Ed:.3f}'
            ))
    else:
        st.warning("Стойността Esr/Ei е извън диапазона на изолиниите за интерполация.")

# Конфигурация на графиката
fig.update_layout(
    xaxis_title="H / D",
    yaxis_title="y",
    legend_title="Изолинии",
    width=900,
    height=600,
    title=f'Комбинирани изолинии: H/D={ratio:.3f}, Esr/En={Esr_over_En:.3f}, En/Ed={En_over_Ed:.3f}',
    hovermode='x unified'
)

# Добавяне на анотации за текущите параметри
fig.add_annotation(
    x=ratio,
    y=0,
    xref="x",
    yref="y",
    text=f"H/D = {ratio:.3f}",
    showarrow=True,
    arrowhead=1,
    ax=0,
    ay=-40
)

st.plotly_chart(fig, use_container_width=True)
