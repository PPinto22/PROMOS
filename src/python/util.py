import MultiNEAT as neat
import csv
import datetime
import json
from collections import namedtuple

import numpy as np


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


def write_summary(out_file_path, best_evaluation, method, substrate=None, **other_info):
    class Results:
        class Network:
            def __init__(self, eval, network=None):
                self.fitness = eval.fitness
                if network is not None:
                    self.connections = get_network_connections(network)
                    self.neurons = get_network_neurons(network)
                    self.neurons_qty = len(self.neurons)
                    self.connections_qty = len(self.connections)

        def __init__(self, best_eval, best_network=None, **other_info):
            self.best = Results.Network(best_eval, best_network)
            for key, value in other_info.items():
                self.__setattr__(key, value)

    net = build_network(best_evaluation.genome, method, substrate)
    results = Results(best_evaluation, net, **other_info)
    with open(out_file_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(results.__dict__, default=serializer, indent=4))


def save_evaluations(out_file_path, gen_evaluations):
    with open(out_file_path, 'w') as file:
        header = ['generation', 'genome_id', 'fitness', 'neurons', 'connections', 'run_minutes']
        writer = csv.writer(file, delimiter=',')
        writer.writerow(header)
        for gen, evaluations in gen_evaluations.items():
            for e in evaluations:
                writer.writerow([gen, e.genome_id, e.fitness, e.neurons, e.connections,
                                 e.global_time.total_seconds()/60.0])
