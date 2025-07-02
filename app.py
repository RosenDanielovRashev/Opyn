import streamlit as st 
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.title("Комбинирани изолинии")

# Зареждане на оригиналните данни
try:
    df_original = pd.read_csv("danni.csv")
    st.write("Оригинални данни (danni.csv):", df_original.head())
except FileNotFoundError:
    st.error("Файлът 'danni.csv' не е намерен")
    df_original = pd.DataFrame()

# Зареждане на новите данни
try:
    df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)  # Преименуване на колоната
    st.write("Нови данни:", df_new.head())
except FileNotFoundError:
    st.error("Файлът 'Оразмеряване на опън за междиннен плстH_D.csv' не е намерен")
    df_new = pd.DataFrame()

# Създаване на фигура
fig = go.Figure()

# 1. Добавяне на оригиналните изолинии за Ei/Ed (от danni.csv)
if not df_original.empty and 'Ei/Ed' in df_original.columns:
    unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
    for level in unique_Ei_Ed:
        df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines',
            name=f'Ei/Ed = {level}',
            line=dict(width=3, color='blue'),
            visible=True
        ))

# 2. Добавяне на новите изолинии за sr_Ei (от новия файл)
if not df_new.empty and 'sr_Ei' in df_new.columns:
    unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
    colors = px.colors.qualitative.Dark2
    for i, sr_Ei in enumerate(unique_sr_Ei):
        df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
        fig.add_trace(go.Scatter(
            x=df_level['H/D'],
            y=df_level['y'],
            mode='lines+markers',
            name=f'σsr/Ei = {sr_Ei}',
            line=dict(width=2, dash='dot', color=colors[i % len(colors)]),
            marker=dict(size=6),
            visible=True
        ))

# Настройки на графиката
fig.update_layout(
    width=800,
    height=800,
    xaxis=dict(
        title='H/D',
        dtick=0.1,
        range=[0, 2],
        constrain='domain',
        showgrid=True,
        gridwidth=1
    ),
    yaxis=dict(
        title='y',
        dtick=0.1,
        range=[0, 2.5],
        scaleanchor='x',
        scaleratio=1,
        showgrid=True,
        gridwidth=1
    ),
    title='Комбинирани изолинии: Ei/Ed (плъсти линии) и σsr/Ei (пунктирани)',
    legend=dict(
        title='Легенда',
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    hovermode='closest'
)

# Бутони за превключване между различните типове изолинии
buttons = [
    dict(label="Всички",
         method="update",
         args=[{"visible": [True]*len(fig.data)}]),
    dict(label="Само Ei/Ed",
         method="update",
         args=[{"visible": [True if trace.name.startswith('Ei/Ed') else False for trace in fig.data]}]),
    dict(label="Само σsr/Ei",
         method="update",
         args=[{"visible": [True if trace.name.startswith('σsr/Ei') else False for trace in fig.data]}])
]

fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        direction="right",
        x=0.5,
        xanchor="center",
        y=1.15,
        yanchor="top",
        buttons=buttons
    )]
)

st.plotly_chart(fig, use_container_width=False)

# Допълнителна информация
st.markdown("""
### Информация за графиката:
- **Сини плъсти линии**: Изолинии за Ei/Ed (от danni.csv)
- **Пунктирани цветни линии**: Изолинии за σsr/Ei (от новия файл)
- **Можете да филтрирате изолиниите** с бутоните над графиката
""")
