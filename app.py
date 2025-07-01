import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    url_ei_ed = "https://raw.githubusercontent.com/<username>/<repo>/main/ei_ed_data.csv"
    url_esr_ei = "https://raw.githubusercontent.com/<username>/<repo>/main/esr_ei_data.csv"

    df_ei_ed = pd.read_csv(url_ei_ed, encoding='utf-8')
    df_esr_ei = pd.read_csv(url_esr_ei, encoding='utf-8')
    return df_ei_ed, df_esr_ei

# Заглавие и легенда
st.title("Оразмеряване на опън в междинен пласт")
st.markdown("""
📌 **Легенда за параметрите:**  
- h: дебелина на пласта [cm]  
- E: модул на еластичност на пласта [MPa]  
- D: диаметър на колелото [cm]  
- Ed: модул на еластичност на подосновата [MPa]  
- Ei: модул на еластичност на междинния пласт [MPa]
""")

df_ei_ed, df_esr_ei = load_data()

# Въвеждане на брой пластове
n = st.number_input("Въведете брой пластове (минимум 2)", min_value=2, step=1)

h_list = []
E_list = []

for i in range(1, n + 1):
    col1, col2 = st.columns(2)
    with col1:
        h = st.number_input(f"h{i} [cm]", min_value=0.0, step=0.1, format="%f")
    with col2:
        E = st.number_input(f"E{i} [MPa]", min_value=0.0, step=1.0, format="%f")
    h_list.append(h)
    E_list.append(E)

if all(v > 0 for v in h_list) and all(v > 0 for v in E_list):
    H = sum(h_list)
    Esr = sum([E_list[i] * h_list[i] for i in range(n - 1)]) / sum(h_list[:-1])

    st.write(f"**Обща дебелина H = {H:.2f} cm**")
    st.write(f"**Изчислен Esr = {Esr:.2f} MPa**")

    Ei = st.number_input("Въведете Ei на междинния пласт [MPa]", min_value=0.0, step=1.0, format="%f")
    Ed = st.number_input("Въведете Ed на подосновата [MPa]", min_value=0.0, step=1.0, format="%f")
    D = st.number_input("Въведете D на колелото [cm]", min_value=0.0, step=0.1, format="%f")

    if Ei > 0 and Ed > 0 and D > 0:
        HD = H / D
        EiEd = Ei / Ed
        EsrEi = Esr / Ei

        st.write(f"H/D = {HD:.3f}")
        st.write(f"Ei/Ed = {EiEd:.3f}")
        st.write(f"Esr/Ei = {EsrEi:.3f}")

        # Визуализация на точката
        fig = px.scatter()
        fig.add_scatter(x=df_ei_ed['H/D'], y=df_ei_ed['σR'],
                        mode='markers', name='Ei/Ed точки', marker=dict(color='blue'))
        fig.add_scatter(x=df_esr_ei['H/D'], y=df_esr_ei['σR'],
                        mode='markers', name='Esr/Ei точки', marker=dict(color='green'))
        fig.add_scatter(x=[HD], y=[0], mode='markers', name='Вашата точка', marker=dict(color='red', size=10))
        fig.update_layout(title="Позиция на точката в номограмата",
                          xaxis_title="H/D",
                          yaxis_title="σR")
        st.plotly_chart(fig)

        st.success("Интерполация на σ може да се добави тук според данните.")
    else:
        st.warning("Моля, въведете Ei, Ed и D с положителни стойности.")
else:
    st.warning("Моля, въведете h и E с положителни стойности за всички пластове.")
