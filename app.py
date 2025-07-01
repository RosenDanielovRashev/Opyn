import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Зареждаме csv файла
df = pd.read_csv('danni.csv')

# Примерен CSV формат:
# y,sigma_R,H/D,Ei/Ed
# 1.0, ..., 0.5, 2.1
# 1.5, ..., 0.7, 3.4
# ...

# За contour plot трябва данните да са под формата на grid (meshgrid).
# Проверяваме уникалните стойности на y и H/D
y_unique = np.sort(df['y'].unique())
h_d_unique = np.sort(df['H/D'].unique())

# Създаваме мрежа
Y, H_D = np.meshgrid(y_unique, h_d_unique)

# Попълваме стойностите на Ei/Ed в съответствие с координатите
# Създаваме матрица със същия размер като мрежата
Z = np.full(Y.shape, np.nan)

# Запълваме Z с данни
for i, h in enumerate(h_d_unique):
    for j, y_val in enumerate(y_unique):
        # Филтрираме реда с текущите координати
        val = df[(df['y'] == y_val) & (df['H/D'] == h)]['Ei/Ed']
        if not val.empty:
            Z[i, j] = val.values[0]

# Създаваме contour plot
fig = go.Figure(data =
    go.Contour(
        z=Z,
        x=y_unique,
        y=h_d_unique,
        colorscale='Viridis',
        contours=dict(
            coloring='lines',  # изолинии
            showlabels=True,  # показване на стойности
            labelfont=dict(size=12, color='white')
        ),
        line_smoothing=0.85,
    )
)

fig.update_layout(
    title='Изолинии на Ei/Ed по y и H/D',
    xaxis_title='y',
    yaxis_title='H/D'
)

fig.show()
