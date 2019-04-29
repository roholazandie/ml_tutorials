import visdom



def simple_line():
    vis = visdom.Visdom()

    trace = dict(x=[1, 2, 3], y=[4, 5, 6], mode="markers+lines", type='custom',
                 marker={'color': 'red', 'symbol': 104, 'size': "10"},
                 text=["one", "two", "three"], name='1st Trace')
    layout = dict(title="First Plot", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})

    #vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})

    vis.line(
        [1,2,3],
        [4,5,6],

        opts=dict(
            xlabel='Step',
            ylabel='Loss',
            title='Loss (mean per 10 steps)',
        )
    )




if __name__ == "__main__":
    simple_line()