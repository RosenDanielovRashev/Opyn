import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="centered")
# Ограничение на ширината със CSS
st.markdown("""
    <style>
    .block-container {
        max-width: 800px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Определяне опънното напрежение в междинен пласт от пътна конструкция")

# --- СЪСТОЯНИЕ ---
if "num_layers" not in st.session_state:
    st.session_state.num_layers = 3
if "current_layer" not in st.session_state:
    st.session_state.current_layer = 0
if "h_values" not in st.session_state:
    st.session_state.h_values = [4.0] * st.session_state.num_layers
if "E_values" not in st.session_state:
    st.session_state.E_values = [1000.0] * st.session_state.num_layers
if "Ed_values" not in st.session_state:
    st.session_state.Ed_values = [100.0] * st.session_state.num_layers
if "D_values" not in st.session_state:
    st.session_state.D_values = [32.04] * st.session_state.num_layers

# --- Брой пластове ---
num_layers = st.number_input(
    "Брой пластове (n)",
    min_value=2, step=1,
    value=st.session_state.num_layers
)

if num_layers != st.session_state.num_layers:
    st.session_state.num_layers = num_layers
    st.session_state.h_values = st.session_state.h_values[:num_layers] + [4.0]*(num_layers - len(st.session_state.h_values))
    st.session_state.E_values = st.session_state.E_values[:num_layers] + [1000.0]*(num_layers - len(st.session_state.E_values))
    st.session_state.Ed_values = st.session_state.Ed_values[:num_layers] + [100.0]*(num_layers - len(st.session_state.Ed_values))
    st.session_state.D_values = st.session_state.D_values[:num_layers] + [32.04]*(num_layers - len(st.session_state.D_values))
    st.session_state.current_layer = min(st.session_state.current_layer, num_layers - 1)

# --- Навигация между пластове ---
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    if st.button("⬅️ Предишен пласт"):
        if st.session_state.current_layer > 0:
            st.session_state.current_layer -= 1
with col3:
    if st.button("Следващ пласт ➡️"):
        if st.session_state.current_layer < st.session_state.num_layers - 1:
            st.session_state.current_layer += 1

layer_idx = st.session_state.current_layer
st.subheader(f"🔧 Данни за пласт {layer_idx + 1} от {num_layers}")

# --- Параметри за текущ пласт ---
D = st.selectbox("Избери D", options=[32.04, 34.0, 33.0],
                 index=[32.04, 34.0, 33.0].index(st.session_state.D_values[layer_idx]), key=f"D_{layer_idx}")
Ed = st.number_input("Ed (MPa)", value=st.session_state.Ed_values[layer_idx], step=0.1, key=f"Ed_{layer_idx}")
h = st.number_input(f"h{layer_idx + 1} (cm):",
                    value=st.session_state.h_values[layer_idx], step=0.1, key=f"h_{layer_idx}")
E = st.number_input(f"E{layer_idx + 1} (MPa):",
                    value=st.session_state.E_values[layer_idx], step=0.1, key=f"E_{layer_idx}")

# --- Запазване на текущите стойности ---
st.session_state.D_values[layer_idx] = D
st.session_state.Ed_values[layer_idx] = Ed
st.session_state.h_values[layer_idx] = h
st.session_state.E_values[layer_idx] = E

# --- Изчисления ---
h_array = np.array(st.session_state.h_values)
E_array = np.array(st.session_state.E_values)

sum_h_n_1 = h_array[:-1].sum()
weighted_sum_n_1 = np.sum(E_array[:-1] * h_array[:-1])
Esr = weighted_sum_n_1 / sum_h_n_1 if sum_h_n_1 else 0

H_n = h_array.sum()
H_n_1 = sum_h_n_1
En = E_array[-1]
Ed_last = st.session_state.Ed_values[-1]
D_last = st.session_state.D_values[-1]

ratio = H_n / D_last if D_last else 0
Esr_over_En = Esr / En if En else 0
En_over_Ed = En / Ed_last if Ed_last else 0

# --- Показване на формули и резултати ---
st.markdown("### 📐 Формули и резултати")
st.latex(r"H_{n-1} = " + " + ".join([f"h_{i+1}" for i in range(len(h_array) - 1)]) +
         f" = {round(H_n_1,3)} cm")
st.latex(r"H_n = " + " + ".join([f"h_{i+1}" for i in range(len(h_array))]) +
         f" = {round(H_n,3)} cm")
st.latex(r"Esr = \frac{\sum E_i h_i}{\sum h_i} = " + f"{round(Esr,3)} MPa")
st.latex(r"\frac{H_n}{D} = " + f"{round(H_n,3)}/{round(D_last,3)} = {round(ratio,3)}")
st.latex(r"E_n = " + f"{round(En,3)} MPa")
st.latex(r"\frac{Esr}{E_n} = " + f"{round(Esr_over_En,3)}")
st.latex(r"\frac{E_n}{Ed} = " + f"{round(En_over_Ed,3)}")

# --- Зареждане на CSV данни ---
try:
    df_original = pd.read_csv("danni.csv")
    df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)
except Exception as e:
    st.error(f"Грешка при зареждане на CSV: {e}")
    st.stop()

# --- Графика ---
fig = go.Figure()

for lvl in sorted(df_original['Ei/Ed'].unique()):
    grp = df_original[df_original['Ei/Ed'] == lvl].sort_values('H/D')
    fig.add_trace(go.Scatter(x=grp['H/D'], y=grp['y'], mode='lines', name=f"Ei/Ed={round(lvl,3)}"))
for lvl in sorted(df_new['sr_Ei'].unique()):
    grp = df_new[df_new['sr_Ei'] == lvl].sort_values('H/D')
    fig.add_trace(go.Scatter(x=grp['H/D'], y=grp['y'], mode='lines', name=f"Esr/Ei={round(lvl,3)}"))

y_point = None
sr_levels = sorted(df_new['sr_Ei'].unique())
if sr_levels[0] <= Esr_over_En <= sr_levels[-1]:
    lows = [lvl for lvl in sr_levels if lvl <= Esr_over_En]
    highs = [lvl for lvl in sr_levels if lvl >= Esr_over_En]
    low = max(lows)
    high = min(highs)
    y_low = np.interp(ratio, df_new[df_new['sr_Ei']==low]['H/D'], df_new[df_new['sr_Ei']==low]['y'])
    y_high = np.interp(ratio, df_new[df_new['sr_Ei']==high]['H/D'], df_new[df_new['sr_Ei']==high]['y'])
    t = (Esr_over_En - low) / (high - low) if high != low else 0
    y_point = y_low + t * (y_high - y_low)
    fig.add_trace(go.Scatter(x=[ratio], y=[y_point], mode='markers',
                             marker=dict(color='red', size=10), name="Резултат"))

fig.update_layout(title="Графика на изолинии", xaxis_title="Hn/D", yaxis_title="y")
st.plotly_chart(fig, use_container_width=True)

# --- Финален резултат σr ---
if y_point is not None:
    sigma_r = round(ratio / 2, 3)
    st.markdown(f"### 🔴 Изчислено σr = {sigma_r}")
else:
    st.warning("⚠️ Няма пресечна точка – σr не може да се определи.")
