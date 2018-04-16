import csv
import datetime
import json
import numpy as np
from collections import namedtuple


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
    # for neuron in network.neurons:
    #     print('af={}, a={}, b={}, bias={}'.format(neuron.activation_function_type, neuron.a, neuron.b, neuron.bias))

    Connection = namedtuple('Connection', 'source target weight')
    return [Connection(c.source_neuron_idx, c.target_neuron_idx, c.weight) for c in network.connections]


def get_current_datetime_string():
    return '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())


def read_data(data_file_path):
    with open(data_file_path) as data_file:
        reader = csv.DictReader(data_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        return list(reader)


class Results:
    class Network:
        def __init__(self, eval):
            self.fitness = eval.fitness
            self.connections = get_network_connections(eval.network) if eval.network is not None else None
            self.neurons = get_network_neurons(eval.network) if eval.network is not None else None
            # self.metrics = eval.metrics

    def __init__(self, run_type, generations, run_time, best, params):
        self.run_type = run_type
        self.generations = generations
        self.run_time = run_time
        self.best = Results.Network(best)
        # self.params = params


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


def write_results(out_file_path, run_type, generations, run_time, best_evaluation, params):
    results = Results(run_type, generations, run_time, best_evaluation, params)
    with open(out_file_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(results.__dict__, default=serializer, indent=4))
