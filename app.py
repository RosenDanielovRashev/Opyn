import pandas as pd
import plotly.express as px

df = pd.read_csv("danni.csv")

fig = px.scatter(
    df,
    x='H/D',
    y='y',
    color='Ei/Ed',
    color_continuous_scale='Viridis',
    title='Scatter plot на Ei/Ed спрямо y и H/D',
    labels={'H/D':'H/D', 'y':'y', 'Ei/Ed':'Ei/Ed'}
)

fig.show()
