import MultiNEAT as neat
import argparse
import datetime
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


def avg(l):
    return sum(l) / len(l)


def time(f, as_microseconds=False):
    pre = datetime.datetime.now()
    f_out = f()
    elapsed_time = datetime.datetime.now() - pre
    if as_microseconds:
        elapsed_time = (elapsed_time.days * 86400 + elapsed_time.seconds) * 10**6 + elapsed_time.microseconds
    return elapsed_time, f_out


def serializer(obj):
    """JSON serializer for objects not serializable by default json code"""

    if any(isinstance(obj, x) for x in (datetime.date, datetime.time)):
        serial = obj.isoformat()
        return serial

    if any(isinstance(obj, x) for x in (datetime.timedelta, np.datetime64)):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj.__dict__


def uint(value):
    def raise_arg_type_error(s):
        raise argparse.ArgumentTypeError("{} is an invalid positive int value".format(value))

    try:
        ivalue = int(value)
        if ivalue <= 0:
            raise_arg_type_error(value)
        return ivalue
    except ValueError:
        raise_arg_type_error(value)


def range_int(value, lower=0, upper=0):
    def raise_arg_type_error(s):
        raise argparse.ArgumentTypeError("{} is not an int value between [{}, {}]".format(value, lower, upper))

    try:
        ivalue = int(value)
        if ivalue < lower or ivalue > upper:
            raise_arg_type_error(value)
        return ivalue
    except ValueError:
        raise_arg_type_error(value)
