import MultiNEAT as neat
import csv
import datetime
import json
from collections import namedtuple

import numpy as np


def get_network_neurons(network):
    Neuron = namedtuple('Neuron', 'index activation_function a b bias')
    return [Neuron(i, str(n.activation_function_type), n.a, n.b, n.bias) for i, n in enumerate(network.neurons)]


def get_network_connections(network):
    Connection = namedtuple('Connection', 'source target weight')
    return [Connection(c.source_neuron_idx, c.target_neuron_idx, c.weight) for c in network.connections]


def get_current_datetime_string():
    return '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())


def build_network(genome, method='neat', substrate=None, **kwargs):
    net = neat.NeuralNetwork()
    if method == 'neat':
        genome.BuildPhenotype(net)
    elif method == 'hyperneat':
        genome.BuildHyperNEATPhenotype(net, substrate)
    else:
        raise ValueError('Invalid method: {}'.format(method))
    return net


def serializer(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime.date) or isinstance(obj, datetime.time):
        serial = obj.isoformat()
        return serial

    if isinstance(obj, datetime.timedelta):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj.__dict__
