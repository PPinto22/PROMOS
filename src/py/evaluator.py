import MultiNEAT as neat
import argparse
import datetime
import multiprocessing
import random
from enum import Enum
from functools import partial

import numpy as np
from sklearn.metrics import roc_curve, auc

import substrate
import util
from data import Data

global_data = None
global_test_data = None


class GenomeEvaluation:
    def __init__(self, genome, fitness, fitness_test=None, neurons=None, connections=None, generation=None,
                 window=None, global_time=None, build_time=None, pred_time=None, pred_avg_time=None, fit_time=None):
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
        self.window = window
        self.global_time = global_time
        self.build_time = build_time
        self.pred_time = pred_time
        self.pred_avg_time = pred_avg_time
        self.fit_time = fit_time
        self.eval_time = sum(x for x in (build_time, pred_time, fit_time) if x is not None)

    def save_genome_copy(self):
        self.genome = neat.Genome(self.genome)


class FitFunction(Enum):
    AUC = 'auc'
    RANDOM = 'random'

    def get_evaluator(self):
        if self is FitFunction.AUC:
            return _evaluate_auc
        elif self is FitFunction.RANDOM:
            return _evaluate_random

    @staticmethod
    def list():
        return list(map(lambda c: c.value, FitFunction))


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


def _evaluate_auc(targets, predictions):
    fpr, tpr, thresholds = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def _evaluate_random(*args):
    fitness = np.random.normal(0.5, 0.1)
    fitness = min(fitness, 1.0)  # Maximum of 1
    fitness = max(fitness, 0.0)  # Minimum of 0
    return fitness


def evaluate(genome, fitfunc, data=None, test_data=None, **kwargs):
    build_time, net = util.time(lambda: util.build_network(genome, **kwargs), as_microseconds=True)

    if data is None:
        data = global_data
    if test_data is None:
        test_data = global_test_data

    evaluator = fitfunc.get_evaluator()

    pred_time, predictions = util.time(lambda: predict(net, data.inputs), as_microseconds=True)
    pred_avg_time = pred_time / len(data)
    fit_time, fitness = util.time(lambda: evaluator(data.targets, predictions), as_microseconds=True)

    predictions_test = predict(net, test_data.inputs) if test_data is not None else None
    fitness_test = evaluator(test_data.targets, predictions_test) if test_data is not None else None

    return _create_genome_evaluation(genome, fitness, net, fitness_test=fitness_test,
                                     build_time=build_time, pred_time=pred_time,
                                     pred_avg_time=pred_avg_time, fit_time=fit_time, **kwargs)


def _create_genome_evaluation(genome, fitness, net=None, fitness_test=None, window=None, generation=None,
                              initial_time=None,
                              build_time=None, pred_time=None, pred_avg_time=None, fit_time=None, **kwargs):
    genome.SetFitness(fitness)
    genome.SetEvaluated()

    global_time = datetime.datetime.now() - initial_time if initial_time is not None else None

    return GenomeEvaluation(genome=genome, fitness=fitness, fitness_test=fitness_test,
                            neurons=net.GetNeuronsQty() if net is not None else None,
                            connections=net.GetConnectionsQty() if net is not None else None,
                            generation=generation, window=window, global_time=global_time, build_time=build_time,
                            pred_time=pred_time, pred_avg_time=pred_avg_time, fit_time=fit_time)


def evaluate_genome_list(genome_list, fitfunc, data, sample_size=0, processes=1, sort=True, test_data=None, **kwargs):
    if sample_size != 0:
        data = data.get_sample(sample_size, seed=random.randint(0, 2147483647))
        if test_data is not None:
            test_data = test_data.get_sample(sample_size, seed=random.randint(0, 2147483647))

    # Set data as a global variables to avoid copies for every process
    global global_data, global_test_data
    global_data = data
    global_test_data = test_data

    evaluator = partial(evaluate, fitfunc=fitfunc, **kwargs)
    if processes == 1:
        evaluation_list = [evaluator(genome) for genome in genome_list]
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            evaluation_list = pool.map(evaluator, genome_list, chunksize=max(len(genome_list) // processes, 1))

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
