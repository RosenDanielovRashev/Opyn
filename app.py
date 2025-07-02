import streamlit as st 
import pandas as pd
import plotly.graph_objects as go

st.title("Комбинирани изолинии")

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Clean column names by removing any leading/trailing whitespace
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Грешка при зареждане на файла {file_path}: {e}")
        return None

# Зареждане на оригиналните данни
df_original = load_data("danni.csv")

# Зареждане на новите данни
df_new = load_data("Оразмеряване на опън за междиннен плстH_D.csv")
if df_new is not None:
    # Clean column names and rename if needed
    if '?sr/Ei' in df_new.columns:
        df_new = df_new.rename(columns={'?sr/Ei': 'sr_Ei'})

# Проверка дали данните са заредени успешно
if df_original is None or df_new is None:
    st.stop()

# Създаване на фигура
fig = go.Figure()

# 1. Добавяне на оригиналните изолинии за Ei/Ed
if 'Ei/Ed' in df_original.columns:
    try:
        unique_Ei_Ed = sorted(df_original['Ei/Ed'].unique())
        for level in unique_Ei_Ed:
            df_level = df_original[df_original['Ei/Ed'] == level].sort_values(by='H/D')
            if not df_level.empty:
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'],
                    y=df_level['y'],
                    mode='lines',
                    name=f'Ei/Ed = {level}',
                    line=dict(width=2)
                )
    except Exception as e:
        st.error(f"Грешка при обработка на оригиналните данни: {e}")

# 2. Добавяне на новите изолинии за sr_Ei
if 'sr_Ei' in df_new.columns:
    try:
        unique_sr_Ei = sorted(df_new['sr_Ei'].unique())
        for sr_Ei in unique_sr_Ei:
            df_level = df_new[df_new['sr_Ei'] == sr_Ei].sort_values(by='H/D')
            if not df_level.empty:
                fig.add_trace(go.Scatter(
                    x=df_level['H/D'],
                    y=df_level['y'],
                    mode='lines',
                    name=f'σsr/Ei = {sr_Ei}',
                    line=dict(width=2, dash='dot')
                )
    except Exception as e:
        st.error(f"Грешка при обработка на новите данни: {e}")

# Проверка дали има добавени данни
if len(fig.data) == 0:
    st.warning("Няма налични данни за визуализация. Моля, проверете имената на колоните във входните файлове.")
    st.stop()

# Настройки на графиката
fig.update_layout(
    width=700,
    height=700,
    xaxis=dict(
        title='H/D',
        dtick=0.1,
        range=[0, 2],
        constrain='domain'
    ),
    yaxis=dict(
        title='y',
        dtick=0.1,
        range=[0, 2.5],
        scaleanchor='x',
        scaleratio=1
    ),
    title='Комбинирани изолинии',
    legend=dict(
        title='Легенда',
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Добавяне на бутон за сваляне на графиката
st.plotly_chart(fig, use_container_width=False)

# Опции за сваляне на данните
st.sidebar.header("Опции за експорт")
if st.sidebar.button("Сваляне на комбинираните данни като CSV"):
    try:
        combined_df = pd.concat([df_original, df_new], axis=0)
        csv = combined_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Натиснете за сваляне",
            data=csv,
            file_name="комбинирани_изолинии.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.sidebar.error(f"Грешка при експортиране на данните: {e}")
