import MultiNEAT as neat
import argparse
import ctypes
import datetime
import multiprocessing as mp
import multiprocessing.sharedctypes
import random
from enum import Enum
from functools import partial

import numpy as np
from sklearn.metrics import roc_curve, auc

import substrate
import util
from bloat import FitnessAdjuster
from data import Data


class GenomeEvaluation:
    def __init__(self, fitness, genome=None, fitness_adj=None, fitness_test=None,
                 genome_neurons=None, genome_connections=None,
                 neurons=None, connections=None, generation=None, window=None,
                 global_time=None, build_time=None, pred_time=None, pred_avg_time=None, fit_time=None):
        self.genome = genome
        self.genome_id = genome.GetID() if genome is not None else -1
        self.fitness = fitness
        self.fitness_adj = fitness if fitness_adj is None else fitness_adj
        self.fitness_test = fitness_test
        self.genome_neurons = genome_neurons
        self.genome_connections = genome_connections
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

    def set_genome(self, genome):
        self.genome = genome
        self.genome_id = genome.GetID()

    def save_genome_copy(self):
        if self.genome is not None:
            self.genome = neat.Genome(self.genome)


class FitFunction(Enum):
    AUC = 'auc'
    RANDOM = 'random'

    def get_evaluator(self):
        if self is FitFunction.AUC:
            return Evaluator._evaluate_auc
        elif self is FitFunction.RANDOM:
            return Evaluator._evaluate_random

    @staticmethod
    def list():
        return list(map(lambda c: c.value, FitFunction))


class Evaluator:
    _inputs, _targets, _test_inputs, _test_targets = [None] * 4
    _pool = None
    multiprocessing = False

    @staticmethod
    def setup(data, test_data=None, processes=1, maxtasksperchild=500):
        Evaluator.close()
        max_size = data.size()
        max_test_size = test_data.size() if test_data is not None else None
        Evaluator._inputs = mp.sharedctypes.RawArray(ctypes.c_double, util.mult(max_size))
        Evaluator._targets = mp.sharedctypes.RawArray(ctypes.c_double, max_size[0])
        Evaluator._test_inputs = mp.sharedctypes.RawArray(ctypes.c_double, util.mult(max_test_size)) \
            if max_test_size is not None else None
        Evaluator._test_targets = mp.sharedctypes.RawArray(ctypes.c_double, max_test_size[0]) \
            if max_test_size is not None else None
        Evaluator.multiprocessing = processes > 1
        Evaluator._pool = mp.Pool(processes=processes, maxtasksperchild=maxtasksperchild) \
            if Evaluator.multiprocessing else None

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def close():
        if Evaluator._pool is not None:
            Evaluator._pool.terminate()
        Evaluator._multiprocessing = False
        Evaluator._inputs, Evaluator._targets, Evaluator._test_inputs, Evaluator._test_targets = [None] * 4

    @staticmethod
    def create_genome_evaluation(genome, fitness, net=None, fitness_test=None, window=None, generation=None,
                                 initial_time=None, build_time=None, pred_time=None, pred_avg_time=None, fit_time=None,
                                 include_genome=False, **kwargs):
        global_time = datetime.datetime.now() - initial_time if initial_time is not None else None

        return GenomeEvaluation(genome=genome if include_genome else None,
                                fitness=fitness, fitness_test=fitness_test,
                                genome_neurons=genome.NumNeurons() if genome is not None else None,
                                genome_connections=genome.NumLinks() if genome is not None else None,
                                neurons=net.GetNeuronsQty() if net is not None else None,
                                connections=net.GetConnectionsQty() if net is not None else None,
                                generation=generation, window=window, global_time=global_time, build_time=build_time,
                                pred_time=pred_time, pred_avg_time=pred_avg_time, fit_time=fit_time)

    @staticmethod
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

    @staticmethod
    def _predict(net, inputs, length, width):
        predictions = np.zeros(length)
        for i in range(length):
            j = i * width
            net.Flush()
            net.Input(inputs[j:j + width])
            net.FeedForward()
            output = net.Output()
            predictions[i] = output[0]
        net.Flush()
        return predictions

    @staticmethod
    def _evaluate_auc(targets, predictions, length=None):
        if length is not None:
            targets = targets[:length]

        fpr, tpr, thresholds = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    @staticmethod
    def _evaluate_random(*args):
        fitness = np.random.normal(0.5, 0.1)
        fitness = min(fitness, 1.0)  # Maximum of 1
        fitness = max(fitness, 0.0)  # Minimum of 0
        return fitness

    @staticmethod
    def _evaluate(genome, fitfunc, size, test_size=None, adjuster=None, **kwargs):
        build_time, net = util.time(lambda: util.build_network(genome, **kwargs), as_microseconds=True)

        evaluator = fitfunc.get_evaluator()

        pred_time, predictions = util.time(lambda: Evaluator._predict(net, Evaluator._inputs, size[0], size[1]),
                                           as_microseconds=True)
        pred_avg_time = pred_time / size[0]
        fit_time, fitness = util.time(lambda: evaluator(Evaluator._targets, predictions, size[0]),
                                      as_microseconds=True)

        predictions_test = Evaluator._predict(net, Evaluator._test_inputs, test_size[0], test_size[1]) \
            if test_size is not None else None
        fitness_test = evaluator(Evaluator._test_targets, predictions_test, test_size[0]) \
            if test_size is not None else None

        evaluation = Evaluator.create_genome_evaluation(genome, fitness, net=net, fitness_test=fitness_test,
                                                        build_time=build_time, pred_time=pred_time,
                                                        pred_avg_time=pred_avg_time, fit_time=fit_time, **kwargs)
        if adjuster is not None:
            evaluation.fitness_adj = adjuster.get_adjusted_fitness(evaluation)

        return evaluation

    @staticmethod
    def evaluate(genome, fitfunc, data, test_data=None, **kwargs):
        size, test_size = Evaluator._set_data(data, test_data)
        evaluation = Evaluator._evaluate(genome, fitfunc, size, test_size, **kwargs)
        genome.SetFitness(evaluation.fitness_adj)
        genome.SetEvaluated()
        evaluation.set_genome(genome)
        return evaluation

    @staticmethod
    def _set_data(data, test_data):
        size = data.size()
        flat_len = util.mult(size)
        assert flat_len <= len(Evaluator._inputs) and size[0] <= len(Evaluator._targets)
        Evaluator._inputs[:flat_len] = data.inputs.ravel()[:flat_len]
        Evaluator._targets[:size[0]] = data.targets[:size[0]]

        test_size = test_data.size() if test_data is not None else None
        if test_size is not None:
            test_flat_len = util.mult(test_size)
            assert test_flat_len <= len(Evaluator._test_inputs) and test_size[0] <= len(Evaluator._test_targets)
            Evaluator._test_inputs[:test_flat_len] = test_data.inputs.ravel()[:test_flat_len]
            Evaluator._test_targets[:test_size[0]] = test_data.targets[:test_size[0]]

        return size, test_size

    @staticmethod
    def evaluate_genome_list(genome_list, fitfunc, data, sample_size=0, sort=True, test_data=None, adjuster=None,
                             **kwargs):
        if sample_size != 0:
            data = data.get_sample(sample_size, seed=random.randint(0, 2147483647))
            if test_data is not None:
                test_data = test_data.get_sample(sample_size, seed=random.randint(0, 2147483647))

        size, test_size = Evaluator._set_data(data, test_data)
        evaluator = partial(Evaluator._evaluate, fitfunc=fitfunc, size=size, test_size=test_size, **kwargs)

        if not Evaluator.multiprocessing:
            evaluation_list = [evaluator(genome) for genome in genome_list]
        else:
            assert Evaluator._pool is not None
            evaluation_list = Evaluator._pool.map(evaluator, genome_list)

        for genome, eval, fitness_adj in zip(genome_list, evaluation_list,
                                             FitnessAdjuster.maybe_get_pop_adjusted_fitness(adjuster, evaluation_list)):
            genome.SetFitness(fitness_adj)
            genome.SetEvaluated()
            eval.fitness_adj = fitness_adj
            eval.set_genome(genome)

        if sort:
            evaluation_list.sort(key=lambda e: e.fitness, reverse=True)

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

    Evaluator.setup(data.size())
    evaluation = Evaluator.evaluate(genome, FitFunction.AUC, data)
    print(evaluation.fitness)
