import MultiNEAT as neat
import argparse
import ctypes
import datetime
import multiprocessing as mp
import multiprocessing.sharedctypes
import concurrent.futures
import random
from enum import Enum
from functools import partial

try:
    import matplotlib.pyplot as plt
except:
    pass

import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score

import substrate
import util
from bloat import FitnessAdjuster
from data import Data
from sliding_window import SlidingWindow

class GenomeEvaluation:
    def __init__(self, fitness, genome=None, fitness_adj=None, fitness_test=None,
                 genome_neurons=None, genome_connections=None,
                 neurons=None, connections=None, generation=None, window=None,
                 global_time=None, build_time=None, pred_time=None, pred_avg_time=None, fit_time=None, **extra):
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
        self.spawn_gen = generation
        self.window = window
        self.global_time = global_time
        self.build_time = build_time
        self.pred_time = pred_time
        self.pred_avg_time = pred_avg_time
        self.fit_time = fit_time
        self.eval_time = sum(x for x in (build_time, pred_time, fit_time) if x is not None)
        for key, value in extra.items():
            self.__setattr__(key, value)

    def set_genome(self, genome):
        self.genome = genome
        self.genome_id = genome.GetID()

    def save_genome_copy(self):
        if self.genome is not None:
            self.genome = neat.Genome(self.genome)


class FitFunction(Enum):
    AUC = 'auc'
    F1 = 'f1'
    RANDOM = 'random'

    def get_evaluator(self):
        if self is FitFunction.AUC:
            return Evaluator._evaluate_auc
        elif self is FitFunction.F1:
            return Evaluator._evaluate_f1
        elif self is FitFunction.RANDOM:
            return Evaluator._evaluate_random

    @staticmethod
    def list():
        return list(map(lambda c: c.value, FitFunction))


class Evaluator:
    _inputs, _targets, _test_inputs, _test_targets = [None] * 4
    _pool = None
    _maxtasks = None
    _tasks = 0
    _processes = 1
    multiprocessing = False

    @staticmethod
    def setup(data, test_data=None, processes=1, maxtasks=10000):
        Evaluator.close()
        max_train_size = data.size()
        max_test_size = test_data.size() if test_data is not None else (0, 0)
        max_size = max_train_size if util.mult(max_train_size) > util.mult(max_test_size) else max_test_size
        Evaluator._inputs = mp.sharedctypes.RawArray(ctypes.c_double, util.mult(max_size))
        Evaluator._targets = mp.sharedctypes.RawArray(ctypes.c_double, max_size[0])
        Evaluator._test_inputs = mp.sharedctypes.RawArray(ctypes.c_double, util.mult(max_test_size)) \
            if test_data is not None else None
        Evaluator._test_targets = mp.sharedctypes.RawArray(ctypes.c_double, max_test_size[0]) \
            if test_data is not None else None
        Evaluator.multiprocessing = processes > 1
        Evaluator._pool = concurrent.futures.ProcessPoolExecutor(max_workers=processes) \
            if Evaluator.multiprocessing else None
        Evaluator._maxtasks = maxtasks
        Evaluator._tasks = 0
        Evaluator._processes = processes

    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def close():
        if Evaluator._pool is not None:
            Evaluator._pool.shutdown()
        Evaluator._multiprocessing = False
        Evaluator._inputs, Evaluator._targets, Evaluator._test_inputs, Evaluator._test_targets = [None] * 4

    @staticmethod
    def create_genome_evaluation(genome, fitness, net=None, fitness_test=None, window=None, generation=None,
                                 initial_time=None, build_time=0, pred_time=0, pred_avg_time=0, fit_time=0,
                                 include_genome=False, extra={}, **kwargs):
        if net is None:
            net = util.build_network(genome, **kwargs)
        global_time = datetime.datetime.now() - initial_time if initial_time is not None else None

        return GenomeEvaluation(genome=genome if include_genome else None,
                                fitness=fitness, fitness_test=fitness_test,
                                genome_neurons=genome.NumHiddenNeurons(),
                                genome_connections=genome.NumLinks(),
                                neurons=net.NumHiddenNeurons(),
                                connections=net.NumConnections(),
                                generation=generation, window=window, global_time=global_time, build_time=build_time,
                                pred_time=pred_time, pred_avg_time=pred_avg_time, fit_time=fit_time, **extra)

    @staticmethod
    def predict_single(net, input):
        net.Flush()
        net.Input(input)
        net.FeedForward()
        return net.Output()[0]

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
    def _evaluate_auc(targets, predictions, length, include_roc=False, **kwargs):
        targets = targets[:length]

        fpr, tpr, thresholds = roc_curve(targets, predictions)
        roc_auc = auc(fpr, tpr)
        roc_auc = util.zero_if_nan(roc_auc)

        if include_roc:
            return roc_auc, {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        else:
            return roc_auc

    @staticmethod
    def _evaluate_f1(targets, predictions, length, **kwargs):
        targets = targets[:length]

        predictions_round = predictions.round()
        f1 = f1_score(targets, predictions_round)
        f1 = util.zero_if_nan(f1)
        return f1

    @staticmethod
    def _evaluate_random(*args, **kwargs):
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
        fit_time, fitness = util.time(lambda: evaluator(Evaluator._targets, predictions, size[0], **kwargs),
                                      as_microseconds=True)
        if isinstance(fitness, tuple):
            fitness = fitness[0]
            extra = fitness[1]
        else:
            extra = {}

        predictions_test = Evaluator._predict(net, Evaluator._test_inputs, test_size[0], test_size[1]) \
            if test_size is not None else None
        fitness_test = evaluator(Evaluator._test_targets, predictions_test, test_size[0]) \
            if test_size is not None else None

        evaluation = Evaluator.create_genome_evaluation(genome, fitness, net=net, fitness_test=fitness_test,
                                                        build_time=build_time, pred_time=pred_time,
                                                        pred_avg_time=pred_avg_time, fit_time=fit_time,
                                                        extra=extra, **kwargs)
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
        try:
            Evaluator._inputs[:flat_len] = data.inputs.ravel()[:flat_len]
            Evaluator._targets[:size[0]] = data.targets[:size[0]]
        except TypeError:
            raise TypeError('Data must be numeric')

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
            Evaluator._update_tasks(len(genome_list))
            futures = [Evaluator._pool.submit(evaluator, genome) for genome in genome_list]
            concurrent.futures.wait(futures)
            evaluation_list = [util.try_(future.result) for future in futures]

        for i, (genome, eval, fitness_adj) in \
                enumerate(zip(genome_list, evaluation_list,
                              FitnessAdjuster.maybe_get_pop_adjusted_fitness(adjuster, evaluation_list))):
            if eval is None:
                eval = Evaluator.create_genome_evaluation(genome, 0, **kwargs)
                evaluation_list[i] = eval
            genome.SetFitness(fitness_adj)
            genome.SetEvaluated()
            eval.fitness_adj = fitness_adj
            eval.set_genome(genome)

        if sort:
            evaluation_list.sort(key=lambda e: e.fitness, reverse=True)

        return evaluation_list

    # This is a workaround for a memory leak -- periodically restarts the processes
    @staticmethod
    def _update_tasks(n):
        if not Evaluator.multiprocessing or Evaluator._maxtasks is None:
            return
        if Evaluator._tasks > Evaluator._maxtasks:
            Evaluator._pool.shutdown()
            Evaluator._pool = concurrent.futures.ProcessPoolExecutor(max_workers=Evaluator._processes)
            Evaluator._tasks = n
        else:
            Evaluator._tasks += n


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('genome_file', help='path to genome file', metavar='GENOME')
    parser.add_argument('data_file', help='path to data file for evaluation', metavar='DATA'),
    methods = ['neat', 'hyperneat', 'eshyperneat']
    parser.add_argument('-m', '--method', dest='method', metavar='M', choices=methods, default='neat',
                        help='which algorithm was used to generate the network: ' + ', '.join(methods))
    parser.add_argument('-e', '--evaluator', dest='evaluator', choices=FitFunction.list(), default='auc',
                        help='evaluation function: ' + ', '.join(FitFunction.list()), metavar='E')
    parser.add_argument('-s', '--substrate', dest='substrate_file', metavar='S', default=None,
                        help='path to a substrate; required if method is hyperneat or eshyperneat')
    parser.add_argument('-W', '--window', dest='width', metavar='W', type=util.uint, default=None,
                        help='Sliding window width (train + test) in hours')
    parser.add_argument('-w', '--test-window', dest='test_width', metavar='W', type=util.uint, default=None,
                        help='Test sliding window width in hours')
    parser.add_argument('-S', '--shift', dest='shift', metavar='S', type=util.uint, default=None,
                        help='Sliding window shift in hours')
    parser.add_argument('--plot', dest='plot', action='store_true', help='Show a visualisation')

    args = parser.parse_args()
    return args


def draw_roc(evaluation, window=None, wait_input=True):
    fig = plt.gcf()
    fig.canvas.set_window_title('ROC Curve {}'.format("(Window {})".format(window)) if window is not None else '')
    plt.clf()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.005])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(evaluation.fpr, evaluation.tpr, color='darkorange',
             lw=2, label='ROC Curve (area = %.2f)' % evaluation.fitness)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve {}'.format("(Window {})".format(window)) if window is not None else '')
    plt.legend(loc="lower right")
    plt.draw()
    if wait_input:
        plt.waitforbuttonpress()
    else:
        plt.pause(0.001)


if __name__ == '__main__':
    args = parse_args()
    genome = neat.Genome(args.genome_file)
    fit_func = FitFunction(args.evaluator)
    data = Data(args.data_file)
    subst = substrate.load_substrate(args.substrate_file) if args.substrate_file is not None else None

    slider = SlidingWindow(args.width, args.shift, args.test_width,
                           file_path=args.data_file) if args.width is not None else None
    if slider is None:
        Evaluator.setup(data)
        evaluation = Evaluator.evaluate(genome, fit_func, data, include_roc=args.plot)
        print(evaluation.fitness)
        if args.plot:
            if fit_func is FitFunction.AUC:
                draw_roc(evaluation)
    else:
        for i, (train_data, test_data) in enumerate(slider):
            Evaluator.setup(test_data)
            evaluation = Evaluator.evaluate(genome, fit_func, test_data, include_roc=args.plot)
            print("Window {}/{}: {}".format(i + 1, slider.n_windows, evaluation.fitness))
            if args.plot:
                if fit_func is FitFunction.AUC:
                    draw_roc(evaluation, window=i + 1, wait_input=i + 1 == slider.n_windows)
