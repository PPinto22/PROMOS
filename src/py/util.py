import MultiNEAT as neat
import argparse
import datetime
import os
from collections import namedtuple

import numpy as np
import pandas as pd

Neuron = namedtuple('Neuron', 'index activation_function a b bias')
Connection = namedtuple('Connection', 'source target weight')


def get_network_neurons(network):
    return [Neuron(i, str(n.activation_function_type), n.a, n.b, n.bias) for i, n in enumerate(network.neurons)]


def get_network_connections(network):
    return [Connection(c.source_neuron_idx, c.target_neuron_idx, c.weight) for c in network.connections]


def get_current_datetime_string():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())


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


def mult(l):
    m = 1
    for v in l:
        m *= v
    return m


def zero_if_nan(x):
    x = x if not np.isnan(x) else 0
    return x


def xor(x, y):
    return bool(x) ^ bool(y)


def get_i(l, i, default=None):
    try:
        return l[i]
    except IndexError:
        return default


def str_to_bool(str):
    if str is True or str is False:
        return str

    if str.lower() in ['yes', 'true', 't', '1']:
        return True
    elif str.lower() in ['no', 'false', 'f', '0']:
        return False
    else:
        raise AttributeError('Invalid boolean string: {}'.format(str))


def soft_sort(to_sort, order):
    val_idx_map = {value: i for i, value in enumerate(to_sort)}
    ret = list(to_sort)
    for i in range(min(len(to_sort), len(order))):
        old, new = ret[i], order[i]
        if new != old and new in val_idx_map:
            new_i = val_idx_map[new]
            ret[i], ret[new_i] = ret[new_i], ret[i]
            val_idx_map[old], val_idx_map[new] = new_i, i
    return ret


def join_str(sep, array):
    res = ''
    for s in array:
        if res == '' and s is not None and s != '':
            res = str(s)
        elif s is not None and s != '':
            res += sep + str(s)
    return res


def list_find(list, key):
    for i, e in enumerate(list):
        if key(e):
            return i
    raise KeyError


def diff_indexes(l1, l2):
    size = min(len(l1), len(l2))
    return [i for i in range(size) if l1[i] != l2[i]]


def table_dict(column):
    if isinstance(column, np.ndarray):
        unique, counts = np.unique(column, return_counts=True)
        return dict(zip(unique, counts))
    elif isinstance(column, pd.Series):
        counts = column.value_counts().to_dict()
        return counts


def make_dir(dir_path=None, file_path=None):
    assert xor(dir_path is not None, file_path is not None)

    if file_path is not None:
        dir_path, _ = os.path.split(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def time(f, as_microseconds=False):
    pre = datetime.datetime.now()
    f_out = f()
    elapsed_time = datetime.datetime.now() - pre
    if as_microseconds:
        elapsed_time = (elapsed_time.days * 86400 + elapsed_time.seconds) * 10 ** 6 + elapsed_time.microseconds
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


def ratio(value):
    def raise_arg_type_error(s):
        raise argparse.ArgumentTypeError("{} is not valid ratio".format(value))

    try:
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
            raise_arg_type_error(value)
        return fvalue
    except ValueError:
        raise_arg_type_error(value)
