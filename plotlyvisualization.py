from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly.offline import plot as offpy
import numpy as np

def countor(X, Y, Z):
    data = [
        go.Contour(
            z=Z,
            x=X,
            y=Y
        )]
    figure1 = dict(data=data)
    offpy(figure1, filename="sample.html", auto_open=True)


def animate_state(all_states):
    trace = go.Heatmap(z=all_states)
    data = [trace]
    figure = dict(data=data)
    offpy(figure, filename="heatmap.html", auto_open=True)





def animate_optimization(X, Y, Z, xs , ys, gradxs, gradys):
    init_notebook_mode(connected=True)
    x = xs
    y = ys
    xm = np.min(x) - 1.5
    xM = np.max(x) + 1.5
    ym = np.min(y) - 1.5
    yM = np.max(y) + 1.5
    N = len(xs)#50
    s = np.linspace(-1, 1, N)
    xx = xs #s + s ** 2
    yy = ys #s - s ** 2

    xxend = xx - gradxs
    yyend = yy - gradys

    data = [
        #go.Contour(z=Z, x=X, y=Y)
        dict(x=x, y=y,
                 name='frame',
                 mode='lines',
                 line=dict(width=2, color='blue')
                 ),
            dict(x=x, y=y,
                 mode='lines',
                 name='curve',
                 line=dict(width=2, color='blue')
                 )
            ]

    layout = dict(xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
                  yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
                  title='Kinematic Generation of a Planar Curve', hovermode='closest',
                  updatemenus=[{'type': 'buttons',
                                'buttons': [{'label': 'Play',
                                             'method': 'animate',
                                             'args': [None]}]}])

    frames1 = [dict(data=[dict(x=[xx[k], xxend[k]],
                              y=[yy[k], yyend[k]],
                              mode='lines',
                              line=dict(color='red', width=2)
                              )
                         ],
                    name=str(k)+"f1") for k in range(N)]

    # frames1 = {'data': [{'x':[xx[k], xxend[k]],
    #                     'y': [yy[k], yyend[k]],
    #                     'mode': 'lines',
    #                     'line':dict(color='red', width=2)
    #                     } for k in range(N)],
    #             'name':'frame1'}

    frames2 = [dict(data=[
                         dict(x=[xx[k]],
                              y=[yy[k]],
                              mode='markers',
                              marker=dict(color='red', size=10)
                              )
                         ],
                    name=str(k)+"f2") for k in range(N)]

    figure1 = {'data': data, 'layout': layout, 'frames': frames1}
    #figure1['frames'].append(frames2)
    #figure1 = dict(data=data, layout=layout, frames=frames)
    offpy(figure1, filename="sample.html", auto_open=True)


def histogram_plot(X):
    data = [go.Histogram(x=X)]
    figure1 = dict(data=data)
    offpy(figure1, filename="histogram.html", auto_open=True)



def plot(X):
    trace = go.Scatter(
        x=np.linspace(0, len(X)),
        y=X,
        name='Low 2007',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=4)
    )
    data = [trace]
    figure = dict(data=data)
    offpy(figure, filename="line_chart.html", auto_open=True)



def plot_surface(X):
    data = [
        go.Surface(
            z=X
        )
    ]
    layout = go.Layout(
        title='',
        autosize=False,
        width=1000,
        height=1000,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        )
    )
    fig = go.Figure(data=data, layout=layout)
    offpy(fig, filename="3dsurface.html", auto_open=True)



if __name__ == "__main__":
    from plotly.offline import init_notebook_mode, iplot
    import numpy as np

    init_notebook_mode(connected=True)
    t=np.linspace(-1,1,100)
    x=t+t**2
    y=t-t**2
    xm=np.min(x)-1.5
    xM=np.max(x)+1.5
    ym=np.min(y)-1.5
    yM=np.max(y)+1.5
    N=50
    s=np.linspace(-1,1,N)
    xx=s+s**2
    yy=s-s**2

    N = 50
    s = np.linspace(-1, 1, N)
    vx = 1 + 2 * s
    vy = 1 - 2 * s  # v=(vx, vy) is the velocity
    speed = np.sqrt(vx ** 2 + vy ** 2)
    ux = vx / speed  # (ux, uy) unit tangent vector, (-uy, ux) unit normal vector
    uy = vy / speed

    xend = xx + ux  # end coordinates for the unit tangent vector at (xx, yy)
    yend = yy + uy



    data = [dict(x=x, y=y,
                 name='frame',
                 mode='lines',
                 line=dict(width=2, color='blue')),
            dict(x=x, y=y,
                 name='curve',
                 mode='lines',
                 line=dict(width=2, color='blue'))
            ]

    layout = dict(width=600, height=600,
                  xaxis=dict(range=[xm, xM], autorange=False, zeroline=False),
                  yaxis=dict(range=[ym, yM], autorange=False, zeroline=False),
                  title='Moving Frenet Frame Along a Planar Curve', hovermode='closest',
                  updatemenus=[{'type': 'buttons',
                                'buttons': [{'label': 'Play',
                                             'method': 'animate',
                                             'args': [None]}]}])

    frames = [dict(data=[dict(x=[xx[k], xend[k]],
                              y=[yy[k], yend[k]],
                              mode='lines',
                              line=dict(color='red', width=2))
                         ]) for k in range(N)]

    figure2 = dict(data=data, layout=layout, frames=frames)
    offpy(figure2)