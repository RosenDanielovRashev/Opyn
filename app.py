import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("Комбинирани изолинии с две X оси")

# Зареждане на данни (замени с реалните си файлове)
try:
    df_original = pd.read_csv("danni.csv")
except Exception:
    st.error("Грешка при зареждане на 'danni.csv'")
    st.stop()

try:
    df_new = pd.read_csv("Оразмеряване на опън за междиннен плстH_D.csv")
    df_new.rename(columns={'Esr/Ei': 'sr_Ei'}, inplace=True)
except Exception:
    st.error("Грешка при зареждане на новия CSV")
    st.stop()

fig = go.Figure()

# Оригинални изолинии спрямо H/D
if 'Ei/Ed' in df_original.columns:
    for level in sorted(df_original['Ei/Ed'].dropna().unique()):
        d = df_original[df_original['Ei/Ed'] == level].sort_values('H/D')
        fig.add_trace(go.Scatter(
            x=d['H/D'], y=d['y'],
            mode='lines',
            name=f'Ei/Ed={level}',
            xaxis='x'
        ))

# Нови изолинии спрямо H/D
if 'sr_Ei' in df_new.columns:
    for level in sorted(df_new['sr_Ei'].dropna().unique()):
        d = df_new[df_new['sr_Ei'] == level].sort_values('H/D')
        fig.add_trace(go.Scatter(
            x=d['H/D'], y=d['y'],
            mode='lines',
            name=f'σsr/Ei={level}',
            line=dict(dash='dot'),
            xaxis='x'
        ))

# Изолинии по σr = H/D / 2, които използват горната ос xaxis2
if 'sr_Ei' in df_new.columns:
    for level in sorted(df_new['sr_Ei'].dropna().unique()):
        d = df_new[df_new['sr_Ei'] == level].sort_values('H/D')
        sigma_r = d['H/D'] / 2
        fig.add_trace(go.Scatter(
            x=sigma_r, y=d['y'],
            mode='lines',
            name=f'σr={level} (горна ос)',
            line=dict(dash='dash'),
            xaxis='x2'
        ))

fig.update_layout(
    width=800,
    height=700,
    title='Комбинирани изолинии с долна ос H/D и горна ос σr',

    # Долна ос
    xaxis=dict(
        title='H/D',
        range=[0, 2],
        domain=[0, 1]
    ),

    # Горна ос (overlay върху долната, но с отделни стойности)
    xaxis2=dict(
        title='σr',
        range=[0, 1],
        side='top',
        overlaying='x',
        showline=True,
        ticks='outside',
        showgrid=False
    ),

    yaxis=dict(
        title='y',
        range=[0, 2.5],
        scaleanchor='x',
        scaleratio=1
    ),

    legend=dict(
        title=dict(text='Легенда'),
        font=dict(size=10)
    )
)

st.plotly_chart(fig, use_container_width=False)
