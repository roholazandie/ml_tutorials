import plotly.graph_objects as go
import plotly
import numpy as np
import plotly.figure_factory as ff
# z = [[.1, .3, .5, .7, .9],
#      [1, .8, .6, .4, .2],
#      [.2, 0, .5, .7, .9],
#      [.9, .8, .4, .2, 0],
#      [.3, .4, .5, .7, 1]]
#
# fig = ff.create_annotated_heatmap(z)
# fig.show()

cm = np.array([[1.5260e+03, 3.4000e+01, 6.0000e+00, 0.0000e+00, 1.0000e+01, 0.0000e+00],
        [2.5000e+01, 1.6200e+02, 4.0000e+00, 0.0000e+00, 1.2000e+01, 1.0000e+00],
        [1.0000e+01, 1.0000e+00, 1.3300e+02, 3.0000e+00, 1.2000e+01, 0.0000e+00],
        [1.0000e+00, 2.0000e+00, 6.0000e+00, 2.2000e+01, 1.3000e+01, 0.0000e+00],
        [9.0000e+00, 2.1000e+01, 8.0000e+00, 4.0000e+00, 1.2200e+02, 0.0000e+00],
        [4.0000e+00, 2.0000e+00, 0.0000e+00, 2.0000e+00, 3.0000e+00, 1.1000e+01]])

cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)
fig = go.Figure(data=go.Heatmap(
                   z=cm,
                   x=["<b>happiness</b>", "<b>surprise</b>", "<b>sadness</b>", "<b>disgust</b>", "<b>anger</b>", "<b>fear</b>"],
                   y=["<b>happiness</b>", "<b>surprise</b>", "<b>sadness</b>", "<b>disgust</b>", "<b>anger</b>", "<b>fear</b>"], colorscale="Jet"),
                layout=go.Layout(
                            width=1000,
                            height=1000,
                        font=dict(family='Helvetica, bold', size=25, color='#000000'),
                        yaxis=dict(autorange='reversed')
                    )

)

fig.show()