import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import plot as offpy
import pandas as pd


def surface(X, Y, Z):
    # Read data from a csv
    #z_data = Z#pd.read_csv('mt_bruno_elevation.csv')
    #print(z_data.as_matrix())
    data = [
        go.Surface(
            x=X,
            y=Y,
            z=Z#z_data.as_matrix()
        )
    ]
    layout = go.Layout(
        title='M',
        autosize=False,
        width=1000,
        height=1000,
    )
    fig = go.Figure(data=data, layout=layout)
    offpy(fig, filename='elevations-3d-surface.html')