import MultiNEAT as neat
import argparse
import datetime
import multiprocessing
import random

import numpy as np
from sklearn.metrics import roc_curve, auc

import substrate
import util
from data import Data

global_data = None
global_test_data = None


class GenomeEvaluation:
    def __init__(self, genome, fitness, fitness_test=None, neurons=None, connections=None, generation=None,
                 global_time=None, **kwargs):
        """
        :type connections: int
        :type neurons: int
        ;type global_time: datetime.timedelta
        """
        self.genome = genome
        self.genome_id = genome.GetID()
        self.fitness = fitness
        self.fitness_test = fitness_test
        self.neurons = neurons
        self.connections = connections
        self.generation = generation
        self.global_time = global_time
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def save_genome_copy(self):
        self.genome = neat.Genome(self.genome)


def predict(net, inputs):
    predictions = np.zeros(len(inputs))
    for i, row in enumerate(inputs):
        net.Flush()
        net.Input(row)
        net.FeedForward()
        output = net.Output()
        predictions[i] = output[0]
    net.Flush()
    return predictions


def _evaluate_auc(net, data):
    predictions = predict(net, data.inputs)
    fpr, tpr, thresholds = roc_curve(data.targets, predictions)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def evaluate_auc(genome, data=None, test_data=None, **kwargs):
    net = util.build_network(genome, **kwargs)

    if data is None:
        data = global_data
    if test_data is None:
        test_data = global_test_data

    fitness = _evaluate_auc(net, data)
    fitness_test = _evaluate_auc(net, test_data) if test_data is not None else None

    return _create_genome_evaluation(genome, fitness, net, fitness_test=fitness_test, **kwargs)


def evaluate_error(genome, data, **kwargs):
    net = util.build_network(genome, **kwargs)

    predictions = predict(net, data.inputs)
    fitness = 1 / sum((abs(pred - target) for pred, target in zip(data.targets, predictions)))

    return _create_genome_evaluation(genome, fitness, net, **kwargs)


def _create_genome_evaluation(genome, fitness, net, fitness_test=None, generation=None, initial_time=None, **kwargs):
    genome.SetFitness(fitness)
    genome.SetEvaluated()

    global_time = datetime.datetime.now() - initial_time if initial_time is not None else None

    return GenomeEvaluation(genome=genome, fitness=fitness, fitness_test=fitness_test,
                            neurons=len(util.get_network_neurons(net)),
                            connections=len(util.get_network_connections(net)),
                            generation=generation, global_time=global_time)


def evaluate_genome_list(genome_list, evaluator, data, sample_size=0, processes=1, sort=True, test_data=None):
    if sample_size != 0:
        data = data.get_sample(sample_size, seed=random.randint(0, 2147483647))
        if test_data is not None:
            test_data = test_data.get_sample(sample_size, seed=random.randint(0, 2147483647))

    # Set data as a global variables to avoid copies for every process
    global global_data, global_test_data
    global_data = data
    global_test_data = test_data

    if processes == 1:
        evaluation_list = [evaluator(genome) for genome in genome_list]
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            evaluation_list = pool.map(evaluator, genome_list, chunksize=len(genome_list) // processes)

        for genome, eval in zip(genome_list, evaluation_list):
            genome.SetFitness(eval.fitness)
            genome.SetEvaluated()

    if sort:
        evaluation_list.sort(key=lambda e: e.fitness, reverse=True)

    global_data = None
    global_test_data = None

    return evaluation_list


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('genome_file', help='path to genome file', metavar='GENOME')
    parser.add_argument('data_file', help='path to data file for evaluation', metavar='DATA'),
    methods = ['neat', 'hyperneat', 'eshyperneat']
    parser.add_argument('-m', '--method', dest='method', metavar='M', choices=methods, default='neat',
                        help='which algorithm was used to generate the network: ' + ', '.join(methods))
    parser.add_argument('-s', '--substrate', dest='substrate_file', metavar='S', default=None,
                        help='path to a substrate; required if method is hyperneat or eshyperneat')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    genome = neat.Genome(args.genome_file)
    data = Data(args.data_file)
    subst = substrate.load_substrate(args.substrate_file) if args.substrate_file is not None else None

    evaluation = evaluate_auc(genome, data)
    print(evaluation.fitness)
