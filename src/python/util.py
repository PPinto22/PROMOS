import MultiNEAT as neat

import csv
import datetime
import json
import numpy as np

from collections import namedtuple
from params import ParametersWrapper


# Get all genomes from the population
def get_genome_list(pop):
    genome_list = []
    for s in pop.Species:
        for i in s.Individuals:
            genome_list.append(i)
    return genome_list


def get_network_neurons(network):
    Neuron = namedtuple('Neuron', 'index activation_function a b bias')
    return [Neuron(i, str(n.activation_function_type), n.a, n.b, n.bias) for i, n in enumerate(network.neurons)]


def get_network_connections(network):
    Connection = namedtuple('Connection', 'source target weight')
    return [Connection(c.source_neuron_idx, c.target_neuron_idx, c.weight) for c in network.connections]


def get_current_datetime_string():
    return '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())


def read_data(data_file_path):
    with open(data_file_path) as data_file:
        reader = csv.DictReader(data_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        return list(reader)


def build_network(genome):
    net = neat.NeuralNetwork()
    genome.BuildPhenotype(net)
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


def write_results(out_file_path, best_evaluation, **other_info):
    class Results:
        class Network:
            def __init__(self, eval, network=None):
                self.fitness = eval.fitness
                if network is not None:
                    self.connections = get_network_connections(network)
                    self.neurons = get_network_neurons(network)
                if eval.metrics is not None:
                    self.metrics = eval.metrics

        def __init__(self, best_eval, best_network=None, **other_info):
            self.best = Results.Network(best_eval, best_network)
            for key, value in other_info.items():
                self.__setattr__(key, value)

    net = build_network(best_evaluation.genome)
    results = Results(best_evaluation, net, **other_info)
    with open(out_file_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(results.__dict__, default=serializer, indent=4))
