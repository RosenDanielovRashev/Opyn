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

# Падащо меню за D
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

# Добавяне на прозрачна линия за втората ос (σₙ)
fig.add_trace(go.Scatter(
    x=np.linspace(0, 1, 50),
    y=[0.05]*50,
    mode='lines',
    line=dict(color='rgba(0,0,0,0)'),
    xaxis='x2',
    showlegend=False,
    hoverinfo='skip'
))

fig.update_layout(
    width=700,
    height=700,
    xaxis=dict(
        title='H/D',
        dtick=0.1,
        range=[0, 2],
        constrain='domain',
        side='bottom',
    ),
    xaxis2=dict(
        title=r'$\sigma_n$',
        overlaying='x',
        side='top',
        range=[0, 1],
        showgrid=False,
        zeroline=False,
        tickvals=np.linspace(0,1,11),
        dtick=0.1
    ),
    yaxis=dict(
        title='y',
        dtick=0.1,
        range=[0, 2.5],
        scaleanchor='x',
        scaleratio=1
    ),
    title='Комбинирани изолинии',
    legend=dict(title='Легенда')
)

# --- Добавяне на точка (Hn/D, Esr/En) чрез интерполация между изолиниите ---

target_Hn_D = ratio
target_sr_Ei = Esr_over_En

unique_sr_Ei = sorted(df_new['sr_Ei'].unique())

lower_sr = max([v for v in unique_sr_Ei if v <= target_sr_Ei], default=unique_sr_Ei[0])
upper_sr = min([v for v in unique_sr_Ei if v >= target_sr_Ei], default=unique_sr_Ei[-1])

df_lower = df_new[df_new['sr_Ei'] == lower_sr].sort_values(by='H/D')
df_upper = df_new[df_new['sr_Ei'] == upper_sr].sort_values(by='H/D')

y_lower = np.interp(target_Hn_D, df_lower['H/D'], df_lower['y'])
y_upper = np.interp(target_Hn_D, df_upper['H/D'], df_upper['y'])

if upper_sr == lower_sr:
    interpolated_y = y_lower
else:
    interpolated_y = y_lower + (y_upper - y_lower) * (target_sr_Ei - lower_sr) / (upper_sr - lower_sr)

fig.add_trace(go.Scatter(
    x=[target_Hn_D],
    y=[interpolated_y],
    mode='markers',
    marker=dict(color='red', size=12, symbol='star'),
    name=f'Точка (Hn/D, Esr/Ei)'
))

st.plotly_chart(fig)

# Показване на резултата под графиката
st.markdown(f"### Точка при Hn/D = {target_Hn_D:.3f} и Esr/Ei = {target_sr_Ei:.3f}")
st.latex(rf"\left({target_Hn_D:.3f}, {interpolated_y:.3f}\right)")
