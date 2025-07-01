import pandas as pd
import plotly.graph_objects as go

# Зареждане на CSV файла
df = pd.read_csv("danni.csv")  # Увери се, че името на файла съвпада

# Pivot таблица: редове = y, колони = H/D, стойности = Ei/Ed
pivot_table = df.pivot_table(index='y', columns='H/D', values='Ei/Ed')

# Създаване на контурна графика
fig = go.Figure(data=
    go.Contour(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis',
        contours=dict(
            coloring='heatmap',
            showlabels=True
        ),
        colorbar=dict(title='Ei/Ed')
    )
)

fig.update_layout(
    title='Контурна графика на Ei/Ed спрямо y и H/D',
    xaxis_title='H/D',
    yaxis_title='y'
)

fig.show()
