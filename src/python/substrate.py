import MultiNEAT as neat

from sympy.geometry import Point


def get_2d_point_line(npoints, point1, point2):
    p1 = Point(point1, evaluate=False)
    p2 = Point(point2, evaluate=False)
    line = []

    if npoints == 1:
        line.append(p1.midpoint(p2))
    else:
        delta = (p2 - p1) / (npoints - 1)
        for i in range(npoints - 1):
            line.append(p1 + delta * i)
        line.append(p2)

    return [(float(p.x), float(p.y)) for p in line]


def init_2d_grid_substrate(inputs: int, hidden_layers: int, nodes_per_layer: [int], outputs: int):
    input_nodes = get_2d_point_line(inputs, (-1.0, -1.0), (-1.0, 1.0))
    hidden_nodes = []
    layers = get_2d_point_line(hidden_layers + 2, (-1, 0), (1, 0))
    for i in range(hidden_layers):
        layer_x = layers[i + 1][0]
        hidden_nodes += get_2d_point_line(nodes_per_layer[i], (layer_x, -1.0), (layer_x, 1.0))
    output_nodes = get_2d_point_line(outputs, (1.0, -1.0), (1.0, 1.0))

    return neat.Substrate(input_nodes, hidden_nodes, output_nodes)


def save_substrate(substrate):
    # TODO
    pass


def load_substrate(substrate_file):
    # TODO
    pass
