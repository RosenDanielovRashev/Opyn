import streamlit as st 
import pandas as pd
import plotly.graph_objects as go

st.title("Изолинии с реален мащаб 1:1 + допълнителна ос σR")

# Зареждане на данни
df = pd.read_csv("danni.csv")

# Уникални стойности
unique_EiEd = sorted(df['Ei/Ed'].unique())
min_HD = df['H/D'].min()
max_HD = df['H/D'].max()

# Входни параметри
selected_EiEd = st.selectbox("Избери Ei/Ed:", unique_EiEd)
selected_HD = st.slider("Избери H/D:", float(min_HD), float(max_HD), step=0.01)

# Изчисляваме y при най-близкия H/D
df_filtered = df[df['Ei/Ed'] == selected_EiEd]
closest_row = df_filtered.iloc[(df_filtered['H/D'] - selected_HD).abs().argmin()]
selected_y = closest_row['y']

# Чертаем изолинии
fig = go.Figure()

for level in unique_EiEd:
    df_level = df[df['Ei/Ed'] == level].sort_values(by='H/D')
    fig.add_trace(go.Scatter(
        x=df_level['H/D'],
        y=df_level['y'],
        mode='lines',
        name=f'Ei/Ed = {level}',
        line=dict(width=2)
    ))

# Добавяме точка
fig.add_trace(go.Scatter(
    x=[selected_HD],
    y=[selected_y],
    mode='markers+text',
    marker=dict(color='red', size=10),
    text=[f"y = {selected_y:.3f}"],
    textposition='top center',
    name="Избрана точка"
))

# Настройки на двете X оси
fig.update_layout(
    width=700,
    height=700,
    xaxis=dict(
        title='H/D',
        dtick=0.2,
        range=[0, 2],
        constrain='domain',
    ),
    xaxis2=dict(
        title='σR',
        overlaying='x',
        side='bottom',
        position=0,  # поставя втората ос под оригиналната
        tickvals=[i for i in range(0, 21)],  # позициите по H/D
        ticktext=[f"{i/20:.2f}" for i in range(0, 21)],  # съответстващи σR от 0 до 1
        showline=True,
        showgrid=False,
        ticks="outside",
    ),
    yaxis=dict(
        title='y',
        dtick=0.1,
        range=[0, 2.7],
        scaleanchor='x',
        scaleratio=1
    ),
    title='Изолинии с реален мащаб 1:1 и допълнителна ос σR',
    legend=dict(title='Легенда')
)

# Показваме графиката
st.plotly_chart(fig, use_container_width=False)

# Пояснение
sigma_R = selected_HD / 2
st.markdown(f"""
**Избрани стойности:**

- Ei/Ed = `{selected_EiEd}`
- H/D = `{selected_HD:.3f}`
- σR = `{sigma_R:.3f}` (изчислено като H/D ÷ 2)
- y = `{selected_y:.4f}`
""")
