import plotly.figure_factory as ff
import numpy as np
import plotly.graph_objects as go


def plot_ternary(X):
    x0 = X[:,0]
    x1 = X[:,1]
    x2 = X[:,2]

    #enthalpy = 2.e6 * (x0 - 0.01) * x2 * (x0 - 0.52) * (x2 - 0.48) * (x1 - 1) ** 2 - 5000
    enthalpy = x0*x1*x2
    fig = ff.create_ternary_contour(np.array([x0, x1, x2]), enthalpy,
                                    pole_labels=['x0', 'x1', 'x2'],
                                    interp_mode='cartesian',
                                    ncontours=20,
                                    colorscale='Viridis',
                                    showscale=True,
                                    title='Mixing enthalpy of ternary alloy')
    fig.show()


def plot_scatter_ternary(X):
    x0 = X[:, 0]
    x1 = X[:, 1]
    x2 = X[:, 2]
    fig = go.Figure(go.Scatterternary(
        text="text",
        a=x0,
        b=x1,
        c=x2,
        mode='markers',
        marker={'symbol': 0,
                'color': 'green',
                'size': 8},
    ))

    fig.update_layout({
        'title': 'Ternary Scatter Plot',
        'showlegend': False
    })

    fig.show()


alpha = [10, 4, 7]
all_pmfs = []
for _ in range(2500):
    pmf = np.random.dirichlet(alpha)
    all_pmfs.append(pmf)

pmfs = np.stack(all_pmfs)
#plot_ternary(pmfs)
plot_scatter_ternary(pmfs)