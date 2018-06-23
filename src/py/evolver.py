#!/usr/bin/python3

import MultiNEAT as neat
import csv
import json
import math
import os
import random
import argparse
import datetime
from sortedcontainers import SortedListWithKey
from functools import partial

import params
import evaluator
import util
from util import avg
import substrate as subst
from data import Data, SlidingWindow


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data_file', help='path to train data file', metavar='DATA'),
    parser.add_argument('-t', '--test', dest='test_file', default=None,
                        help='path to test data file', metavar='FILE')
    parser.add_argument('-o', '--outdir', dest='out_dir', default='.',
                        help='directory where to save results. If NULL, do not save results.', metavar='DIR')
    methods = ['neat', 'hyperneat', 'eshyperneat']
    parser.add_argument('-m', '--method', dest='method', choices=methods, default='neat',
                        help='which algorithm to run: ' + ', '.join(methods), metavar='M')
    parser.add_argument('-P', '--params', dest='params', metavar='FILE', default=None,
                        help='path to a parameters file. If not specified, default values will be used.')
    parser.add_argument('-x', '--substrate', dest='substrate', metavar='X', default=0,
                        type=partial(util.range_int, lower=0, upper=len(subst.substrates) - 1),
                        help='which substrate to use, 0 <= X <= {}'.format(len(subst.substrates) - 1))
    parser.add_argument('-e', '--evaluator', dest='evaluator', choices=evaluator.FitFunction.list(), default='auc',
                        help='evaluation function: ' + ', '.join(evaluator.FitFunction.list()), metavar='E')
    parser.add_argument('-g', '--generations', dest='generations', type=util.uint, metavar='G', default=None,
                        help='number of generations per run or, if the option -W is specified, per sliding window')
    parser.add_argument('-T', '--time', dest='time_limit', type=util.uint, metavar='MIN', default=None,
                        help='time limit (in minutes) per run or, if the option -W is specified, per sliding window')
    parser.add_argument('-p', '--processes', dest='processes', type=util.uint, default=1,
                        help='number of processes to use for parallel evaluation. '
                             'If P=1, the evaluations will be sequential', metavar='P')
    parser.add_argument('-s', '--sample', dest='sample_size', type=util.uint, default=0, metavar='N',
                        help='use a balanced sample of size N in evaluations. If S=0, use the whole data-set')
    parser.add_argument('-l', '--load', dest='pop_file', metavar='FILE', default=None,
                        help='load the contents of FILE as the initial population and parameters')
    parser.add_argument('-r', '--runs', dest='runs', metavar='R', help='run R times', type=util.uint, default=1)
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
                        help='Do not print any messages to stdout, except for the result')
    parser.add_argument('--id', dest='id', metavar='ID', default=None,
                        help='run identifier. This ID will be used to name all output files '
                             '(e.g., neat_ID_summary.txt). '
                             'If unspecified, the ID will be the datetime of when the run was started.')
    parser.add_argument('--seed', dest='seed', metavar='S', type=util.uint, default=None,
                        help='specify an RNG integer seed')
    parser.add_argument('--test-fitness', dest='test_fitness', action='store_true',
                        help='evaluate every individual with a sample (of size N=sample_size) of the test data-set. '
                             'These evaluations are for validation only, and have no influence over the evolutionary'
                             ' process. This option slows down the execution speed by half.')
    parser.add_argument('--no-statistics', dest='no_statistics', action='store_true',
                        help='do not record any statistics regarding the progress of individuals over time')
    parser.add_argument('--no-reevaluation', dest='no_reevaluation', action='store_true',
                        help='applicable if a sample size is specified. '
                             'If set, there will be no final reevaluation of '
                             'the best individuals with the whole data-set.')
    parser.add_argument('-W', '--window', dest='width', metavar='W', type=util.uint, default=None,
                        help='Sliding window width (train + test) in hours')
    parser.add_argument('-w', '--test-window', dest='test_width', metavar='W', type=util.uint, default=None,
                        help='Test sliding window width in hours')
    parser.add_argument('-S', '--shift', dest='shift', metavar='S', type=util.uint, default=None,
                        help='Sliding window shift in hours')

    options = parser.parse_args()

    options.id = options.id if options.id is not None else util.get_current_datetime_string()
    options.out_dir = options.out_dir if options.out_dir != 'NULL' else None

    if options.seed is not None:
        random.seed(options.seed)

    return options


class Summary:
    class Best:
        def __init__(self, eval, test, network=None):
            self.fitness = eval.fitness
            self.fitness_test = test
            if network is not None:
                self.connections = util.get_network_connections(network)
                self.neurons = util.get_network_neurons(network)
                self.neurons_qty = len(self.neurons)
                self.connections_qty = len(self.connections)

    def __init__(self, best_eval, fitness_test, best_network=None, **other_info):
        self.best = Summary.Best(best_eval, fitness_test, best_network)
        for key, value in other_info.items():
            self.__setattr__(key, value)


class Evolver:
    def __init__(self, options):
        self.options = options

        # Sliding window assertions
        assert not all(x is not None for x in (self.options.test_file, self.options.test_width)), \
            'Either specify a static test file (-t) or a test sliding window width (-w); not both'
        assert all(x is not None for x in (self.options.width, self.options.shift)) or \
               not any(x is not None for x in (self.options.width, self.options.shift)), \
            'Both window width (-W) and window shift (-S) are required'
        if self.options.test_width is not None:
            assert self.options.width is not None, 'The test width option (-w) requires the window width option (-W)'

        # Evaluation function
        self.fitness_func = evaluator.FitFunction(self.options.evaluator)

        # MultiNEAT parameters
        self.params = params.get_params(self.options.params)

        # Sliding window
        self.width = self.options.width if self.options.width is not None else 0
        self.test_width = self.options.test_width if self.options.test_width is not None else 0
        self.shift = self.options.shift if self.options.shift is not None else 0
        self.is_online = self.options.width is not None  # Is sliding window being used
        self.slider = SlidingWindow(self.width, self.shift, self.test_width, file_path=self.options.data_file) \
            if self.is_online else None

        # Data
        if self.is_online:  # Set the first window
            assert self.slider.has_next(), 'The specified window width (-W) is too large for the available data'
            self.train_data, test_data = next(self.slider)
            self.test_data = Data(self.options.test_file) if self.options.test_file is not None else test_data
        else:  # Use static data
            self.train_data = Data(self.options.data_file)
            self.test_data = Data(self.options.test_file) if self.options.test_file is not None else None

        # Substrate for HyperNEAT and ES-HyperNEAT
        try:
            self.substrate = subst.get_substrate(self.options.substrate,
                                                 inputs=self.train_data.n_inputs,
                                                 hidden_layers=10, nodes_per_layer=[10] * 10,
                                                 outputs=1) \
                if self.options.method in ['hyperneat', 'eshyperneat'] else None
        except IndexError:
            raise ValueError('Invalid substrate choice: {} (should be 0 <= X <= {})'.
                             format(self.options.substrate, len(subst.substrates) - 1)) from None

        self.initial_time = None         # When the run started
        self.window_initial_time = None  # When the current window started
        self.eval_time = None            # Time spent in evaluations
        self.window_eval_time = None     # Time spent in evaluations during the current window
        self.ea_time = None              # Time spent in the EA
        self.window_ea_time = None       # Time spent in the EA during the current window

        self.run_i = None  # Current run, in case of multiple runs

        self.pop = self.init_population()  # C++ Population
        self.params = self.pop.Parameters  # Needed in case pop is loaded from file
        self.generation = 0  # Current generation
        # All time best evaluations, ordered from best to worst fitness
        self.best_list = SortedListWithKey(key=lambda x: -x.fitness)
        self.best_set = set()  # Set of IDs of the individuals in best_list
        self.best_test = None  # GenomeEvaluation (evaluated with the test data-set) of the best individual in best_test

    def clear(self):
        self.initial_time = None
        self.eval_time = datetime.timedelta()
        self.ea_time = datetime.timedelta()
        if self.is_online:
            self.slider.reset()
            self.train_data, test = next(self.slider)
            if self.test_width is not None:
                self.test_data = test
        self.run_i = None
        self.pop = self.init_population()
        self.generation = 0
        self.best_list.clear()
        self.best_set.clear()
        self.best_test = None

    def init_population(self):
        if self.options.pop_file is not None:
            return neat.Population(self.options.pop_file)

        pop = None
        output_act_f = neat.ActivationFunction.UNSIGNED_SIGMOID
        hidden_act_f = neat.ActivationFunction.UNSIGNED_SIGMOID
        seed = random.randint(0, 2147483647) if self.options.seed is None else self.options.seed

        if self.options.method == 'neat':
            g = neat.Genome(0, self.train_data.get_num_inputs(), 0, 1, False, output_act_f, hidden_act_f, 0,
                            self.params, 0)
            pop = neat.Population(g, self.params, True, 1.0, seed)
        elif self.options.method in ['hyperneat', 'eshyperneat']:
            g = neat.Genome(0, self.substrate.GetMinCPPNInputs(), 0, self.substrate.GetMinCPPNOutputs(),
                            False, output_act_f, hidden_act_f, 0, self.params, 0)
            pop = neat.Population(g, self.params, True, 1.0, seed)

        return pop

    def get_genome_list(self):
        return [individual for species in self.pop.Species for individual in species.Individuals]

    def elapsed_time(self):
        return datetime.datetime.now() - self.initial_time

    def window_elapsed_time(self):
        return datetime.datetime.now() - self.window_initial_time

    def is_finished(self):
        if not self.is_online:
            return self.is_window_finished()  # Assuming no sliding window as equivalent to a single window
        else:
            # It's the last window and it is has finished
            return not self.slider.has_next() and self.is_window_finished()

    def print(self, msg, override=False):
        if not self.options.quiet or override:
            print(msg)

    def make_out_dir(self):
        if self.options.out_dir is None:
            raise ValueError('out_dir is None')

        if not os.path.exists(self.options.out_dir):
            os.makedirs(self.options.out_dir)

    def get_out_file_path(self, suffix, include_window=True):
        if self.options.out_dir is None:
            raise ValueError('out_dir is None')

        run = '({})'.format(self.run_i) if self.run_i is not None else ''
        window = '({})'.format(self.get_current_window()) if include_window and self.is_online else ''

        # Format: '<OUTDIR>/<METHOD>_<ID><(RUN_INDEX)><(WINDOW_INDEX)>_<SUFFIX>
        # Example: 'results/neat_sample1K(0)(0)_summary.json'
        return '{}/{}_{}{}{}_{}'.format(self.options.out_dir, self.options.method, self.options.id, run, window, suffix)

    def write_results(self):
        if self.options.out_dir is not None:
            self.make_out_dir()
            self.save_window_summary()
            self.write_summary(self.get_out_file_path('summary.json'))
            self.pop.Save(self.get_out_file_path('population.txt'))
            self.get_best().genome.Save(self.get_out_file_path('best.txt'))

    def get_summary(self):
        best_evaluation = self.get_best()
        net = util.build_network(best_evaluation.genome, self.options.method, self.substrate)
        date_begin, date_end = self.train_data.get_time_range()
        return Summary(
            best_eval=best_evaluation, best_network=net,
            fitness_test=self.best_test.fitness if self.best_test is not None else None,
            # Other info
            params=params.ParametersWrapper(self.params), generations=self.generation,
            run_time=datetime.datetime.now() - self.initial_time, eval_time=self.eval_time,

            ea_time=self.ea_time, processes=self.options.processes,
            sample_size=self.options.sample_size if self.options.sample_size is not None else len(self.train_data),
            window=self.get_current_window() if self.is_online else None,
            date_begin=date_begin, date_end=date_end, train_size=len(self.train_data),
            train_positives=len(self.train_data.positives), train_negatives=len(self.train_data.negatives),
            test_size=len(self.test_data) if self.test_data is not None else -1,
            test_positives=len(self.test_data.positives) if self.test_data is not None else -1,
            test_negatives=len(self.test_data.negatives) if self.test_data is not None else -1
        )

    def write_summary(self, file_path):
        with open(file_path, 'w', encoding='utf8') as f:
            f.write(json.dumps(self.get_summary().__dict__, default=util.serializer, indent=4))

    def save_window_summary(self):
        if not self.is_online or self.options.out_dir is None:
            return

        self.make_out_dir()
        file_path = self.get_out_file_path('windows.csv', include_window=False)
        if self.get_current_window() == 0:
            with open(file_path, 'w') as file:
                writer = csv.writer(file, delimiter=',')
                header = ['window', 'begin_date', 'end_date', 'generations', 'run_time', 'eval_time', 'ea_time',
                          'train_size', 'train_positives', 'train_negatives',
                          'test_size', 'test_positives', 'test_negatives',
                          'train_fitness', 'test_fitness',
                          'best_neurons', 'best_connections', ]
                writer.writerow(header)
        with open(file_path, 'a') as file:
            writer = csv.writer(file, delimiter=',')
            test_size = len(self.test_data) if self.test_data is not None else -1
            test_positives = len(self.test_data.positives) if self.test_data is not None else -1
            test_negatives = len(self.test_data.negatives) if self.test_data is not None else -1
            test_fitness = self.best_test.fitness if self.best_test is not None else -1
            best = self.get_best()
            begin_date, end_date = self.train_data.get_time_range()
            writer.writerow([self.get_current_window(), begin_date, end_date, self.generation,
                             self.window_elapsed_time().total_seconds() / 60.0,
                             self.window_eval_time.total_seconds() / 60.0,
                             self.window_ea_time.total_seconds() / 60.0,
                             len(self.train_data), len(self.train_data.positives), len(self.train_data.negatives),
                             test_size, test_positives, test_negatives, best.fitness, test_fitness,
                             best.neurons, best.connections])

    def save_evaluations(self, evaluations):
        if self.options.no_statistics or self.options.out_dir is None:
            return

        self.make_out_dir()
        file_path = self.get_out_file_path('evaluations.csv', include_window=False)
        if self.generation == 0:
            with open(file_path, 'w') as file:
                writer = csv.writer(file, delimiter=',')
                header = ['window', 'generation', 'genome_id', 'fitness', 'fitness_test', 'neurons', 'connections',
                          'build_time', 'pred_time', 'pred_avg_time', 'fit_time', 'run_time']
                writer.writerow(header)
        with open(file_path, 'a') as file:
            writer = csv.writer(file, delimiter=',')
            for e in evaluations:
                fitness_test = e.fitness_test if e.fitness_test is not None else -1
                writer.writerow([e.window, e.generation, e.genome_id, e.fitness, fitness_test, e.neurons, e.connections,
                                 e.build_time, e.pred_time, e.pred_avg_time, e.fit_time,
                                 e.global_time.total_seconds() / 60.0])

    def get_best(self):
        return self.best_list[0]

    def update_best_list(self, evaluations):
        max_updates = math.ceil(0.05 * self.params.PopulationSize)  # Take at most the best 5% of evaluations

        # Evaluations must be sorted by descending fitness
        for i in range(max_updates):
            e = evaluations[i]
            # Break condition (best_list is full and e is worse than the worst evaluation in best_list)
            if len(self.best_list) == self.params.PopulationSize and e.fitness < self.best_list[-1].fitness:
                break
            elif e.genome_id in self.best_set:  # Individual already exists in best_list; skip
                continue
            else:  # Add to best_list
                index = self._add_to_best_list(e)

                self.print("New best (#{})> Fitness: {:.6f}, Neurons: {}, Connections:{}".
                           format(index, e.fitness, e.neurons, e.connections))

                # Cap the size of best_list at PopulationSize
                if len(self.best_list) > self.params.PopulationSize:
                    self._remove_from_best_list(-1)

    def _add_to_best_list(self, evaluation):
        evaluation.save_genome_copy()
        i = self.best_list.bisect_left(evaluation)
        self.best_list.insert(i, evaluation)
        self.best_set.add(evaluation.genome_id)
        return i

    def _remove_from_best_list(self, index):
        self.best_set.remove(self.best_list[index].genome_id)
        del self.best_list[index]

    def reevaluate_best_list(self):
        evaluation_list = self.evaluate_list([e.genome for e in self.best_list], sample_size=0)
        self.best_list.clear()
        for e in evaluation_list:
            self.best_list.add(e)

    def evaluate_list(self, genome_list, sample_size=None):
        sample_size = sample_size if sample_size is not None else self.options.sample_size
        test_data = self.test_data if self.options.test_fitness and not self.options.no_statistics else None
        return evaluator.evaluate_genome_list(
            genome_list, self.fitness_func, data=self.train_data,
            sample_size=sample_size, processes=self.options.processes, test_data=test_data,
            # Extra **kwargs
            method=self.options.method, substrate=self.substrate,
            generation=self.generation, window=self.get_current_window(), initial_time=self.initial_time
        )

    def evaluate(self, genome):
        return self._evaluate(genome, self.train_data)

    def evaluate_test(self, genome):
        return self._evaluate(genome, self.test_data)

    def _evaluate(self, genome, data):
        return evaluator.evaluate(genome, self.fitness_func, data, method=self.options.method, substrate=self.substrate,
                                     generation=self.generation, window=self.get_current_window(),
                                     initial_time=self.initial_time)

    def evaluate_pop(self):
        pre_eval_time = datetime.datetime.now()
        evaluation_list = self.evaluate_list(self.get_genome_list())
        time_diff = datetime.datetime.now() - pre_eval_time
        self.eval_time += time_diff
        self.window_eval_time += time_diff

        self.save_evaluations(evaluation_list)
        self.update_best_list(evaluation_list)

    def epoch(self):
        pre_ea_time = datetime.datetime.now()
        self.pop.Epoch()
        time_diff = datetime.datetime.now() - pre_ea_time
        self.ea_time += time_diff
        self.window_ea_time += time_diff
        self.generation += 1

    def print_best(self):
        best = self.get_best()
        best_test_str = ' Fitness (test): {:.6f},'.format(self.best_test.fitness) if self.best_test is not None else ''

        self.print("\nBest result> Fitness (train): {:.6f},{} Neurons: {}, Connections: {}".
                   format(best.fitness, best_test_str, best.neurons, best.connections), override=True)

    def evaluate_best_test(self):
        best = self.get_best()
        if self.test_data is not None:
            self.best_test = self.evaluate_test(best.genome)

    def termination_sequence(self):
        # Reevaluate the best individuals with full data if sample_size is specified
        if self.options.sample_size != 0 and not self.options.no_reevaluation:
            self.print("\nReevaluating the best individuals with the whole data-set...")
            self.reevaluate_best_list()

        self.evaluate_best_test()  # Test the best individual obtained with the test data-set
        self.print_best()  # Print to stdout the best result
        self.write_results()  # Write run details to files

    def shift_window(self):
        self.termination_sequence()

        self.best_list.clear()
        self.best_set.clear()
        self.best_test = None
        self.reset_window_timers()

        self.train_data, test = next(self.slider)
        if test is not None:
            self.test_data = test

    def should_shift(self):
        if not self.is_online or not self.slider.has_next():
            return False
        return self.is_window_finished()

    def is_window_finished(self):
        window = self.get_current_window() + 1

        generation_limit = self.options.generations is not None and self.generation >= self.options.generations * window
        time_limit = self.options.time_limit is not None and \
                     self.elapsed_time().total_seconds() / 60 >= self.options.time_limit * window
        return generation_limit or time_limit

    def get_current_window(self):
        return self.slider.get_current_window_index() if self.slider is not None else 0

    def init_timers(self):
        self.initial_time = datetime.datetime.now()
        self.ea_time = datetime.timedelta()
        self.eval_time = datetime.timedelta()
        self.reset_window_timers()

    def reset_window_timers(self):
        self.window_initial_time = datetime.datetime.now()
        self.window_ea_time = datetime.timedelta()
        self.window_eval_time = datetime.timedelta()

    def _run(self):
        self.init_timers()

        # Run the EA
        while not self.is_finished():
            window_info = '[Window {}/{}] '.format(self.get_current_window() + 1,
                                                   self.slider.n_windows) if self.is_online else ''
            self.print("\n{}Generation {} ({})".format(window_info, self.generation, self.elapsed_time()))
            self.evaluate_pop()
            self.epoch()

            if self.should_shift():
                self.shift_window()

        self.termination_sequence()

    def _multiple_runs(self):
        class Summary:
            def __init__(self, runs_list):
                best_run_i, best_run = max(enumerate(runs_list), key=lambda x: x[1].best.fitness)
                self.best_run = best_run_i
                self.best_fitness = best_run.best.fitness
                self.average_fitness = avg([run.best.fitness for run in runs_list])

                if best_run.best.fitness_test is not None:
                    self.best_fitness_test = best_run.best.fitness_test
                    self.average_fitness_test = avg([run.best.fitness_test for run in runs_list])

                run_time_list = [run.run_time for run in runs_list]
                self.average_run_time = sum(run_time_list, datetime.timedelta()) / len(run_time_list)
                self.total_run_time = sum(run_time_list, datetime.timedelta())

        runs_summary_list = []
        for i in range(self.options.runs):
            self.run_i = i
            self._run()
            runs_summary_list.append(self.get_summary())
            self.clear()

        with open(self.get_out_file_path('summary.json'), 'w', encoding='utf8') as f:
            f.write(json.dumps(Summary(runs_summary_list).__dict__, default=util.serializer, indent=4))

    def run(self):
        if self.options.runs > 1:
            self._multiple_runs()
        else:
            self._run()


if __name__ == '__main__':
    options = parse_args()
    evolver = Evolver(options)
    evolver.run()
