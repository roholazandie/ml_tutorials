from fastdot import *
from fastcore.all import *
from graphviz import Source


# g = Dot()
# c = Cluster('cl', fillcolor='pink')
# a1, a2, b = c.add_items('a', 'a', 'b')
# c.add_items(a1.connect(a2), a2.connect(b))
# g.add_item(Node('Check tooltip', tooltip="I have a tooltip!"))
# g.add_item(c)
#
# s = Source(g, filename="test.gv", format="png")
# s.view()

def neuralnet_view():
    @dataclass(frozen=True)
    class Layer:
        name: str
        n_filters: int = 1

    class Linear(Layer):
        pass

    class Conv2d(Layer):
        pass

    @dataclass(frozen=True)
    class Sequential:
        layers: list
        name: str

    block1 = Sequential([Conv2d('conv', 5), Linear('lin', 3)], 'block1')
    block2 = Sequential([Conv2d('conv1', 8), Conv2d('conv2', 2), Linear('lin')], 'block2')

    node_defaults['fillcolor'] = lambda o: 'greenyellow' if isinstance(o, Linear) else 'pink'
    cluster_defaults['label'] = node_defaults['label'] = attrgetter('name')
    node_defaults['tooltip'] = str

    c1 = seq_cluster(block1.layers, block1)
    c2 = seq_cluster(block2.layers, block2)
    e1, e2 = c1.connect(c2), c1.connect(c2.last())
    g = graph_items(c1, c2, e1, e2)

    s = Source(g, filename="test.gv", format="png")
    s.view()


def my_example():
    @dataclass(frozen=True)
    class Sequence:
        name: str
        subseqeunce: list

    @dataclass(frozen=True)
    class Atom1:
        name: str

    @dataclass(frozen=True)
    class Atom2:
        name: str
        number: int

    block1 = Sequence('block1', [Atom1('a1'), Atom2('a2', 3)])
    block2 = Sequence('block2', [Atom2('a3', 5), Atom2('a4', 5), Atom1('a5')])

    def color_map(o):
        if isinstance(o, Atom1):
            return 'green'
        elif isinstance(o, Atom2):
            return 'blue'
        elif isinstance(o, Sequence):
            return 'pink'

    node_defaults['fillcolor'] = color_map
    cluster_defaults['label'] = node_defaults['label'] = attrgetter('name')
    node_defaults['tooltip'] = str

    c1 = seq_cluster(block1.subseqeunce, block1)
    c2 = seq_cluster(block2.subseqeunce, block2)
    e1 = c1.connect(c2)
    e2 = c1.connect(c2[1])
    e3 = c1[0].connect(c2[1])
    g = graph_items(c1, c2, e1, e2, e3)
    s = Source(g, filename="test.gv", format="png")
    s.view()

if __name__ == "__main__":
    #neuralnet_view()
    my_example()