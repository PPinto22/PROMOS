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
from data import Data


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
    parser.add_argument('-S', '--substrate', dest='substrate', type=int, metavar='S', default=0,
                        help='which substrate to use, 0 <= S <= {}'.format(len(subst.substrates) - 1))
    evaluation_functions = ['auc']
    parser.add_argument('-e', '--evaluator', dest='evaluator', choices=evaluation_functions, default='auc',
                        help='evaluation function: ' + ', '.join(evaluation_functions), metavar='E')
    parser.add_argument('-g', '--generations', dest='generations', type=int, default=None,
                        help='number of generations', metavar='G')
    parser.add_argument('-T', '--time', dest='time_limit', type=int, default=None,
                        help='time limit in minutes', metavar='MIN')
    parser.add_argument('-p', '--processes', dest='processes', type=int, default=1,
                        help='number of processes to use for parallel evaluation. '
                             'If P=1, the evaluations will be sequential', metavar='P')
    parser.add_argument('-s', '--sample', dest='sample_size', type=int, default=0, metavar='N',
                        help='use a balanced sample of size N in evaluations. If S=0, use the whole data-set')
    parser.add_argument('-l', '--load', dest='pop_file', metavar='FILE', default=None,
                        help='load the contents of FILE as the initial population and parameters')
    parser.add_argument('-r', '--runs', dest='runs', metavar='R', help='run R times', type=int, default=1)
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true',
                        help='Do not print any messages to stdout, except for the result')
    parser.add_argument('--id', dest='id', metavar='ID', default=None,
                        help='run identifier. This ID will be used to name all output files '
                             '(e.g., neat_ID_summary.txt). '
                             'If unspecified, the ID will be the datetime of when the run was started.')
    parser.add_argument('--seed', dest='seed', metavar='S', type=int, default=None,
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
    parser.add_argument('-W', '--window', dest='window', metavar='W', default=None,
                        help='TODO')
    parser.add_argument('-w', '--test-window', dest='test_width', metavar='W', default=None,
                        help='TODO')
    parser.add_argument('-f', '--frequency', dest='frequency', metavar='F', default=None,
                        help='TODO')

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

        # Evaluation function
        if self.options.evaluator == 'auc':
            self.genome_evaluator = evaluator.evaluate_auc
        else:
            raise ValueError('Invalid genome evaluator: {}'.format(self.options.evaluator))

        # MultiNEAT parameters
        self.params = params.get_params(self.options.params)
        # Substrate for HyperNEAT and ES-HyperNEAT
        try:
            self.substrate = subst.get_substrate(self.options.substrate) if \
                self.options.method in ['hyperneat', 'eshyperneat'] else None
        except IndexError:
            raise ValueError('Invalid substrate choice: {} (should be 0 <= S <= {})'.
                             format(self.options.substrate, len(subst.substrates) - 1)) from None

        self.data = Data(self.options.data_file)  # Training data
        self.test_data = Data(self.options.test_file) if self.options.test_file is not None else None  # Test data

        self.initial_time = None  # Time when the run starts
        self.eval_time = datetime.timedelta()  # Time spent in evaluations
        self.ea_time = datetime.timedelta()  # Time spent in the EA

        self.cur_run = None  # Current run, in case of multiple runs
        self.cur_window = None  # Current sliding window index, in case of online training

        self.pop = self.init_population()  # C++ Population
        self.params = self.pop.Parameters  # Needed in case pop is loaded from file
        self.generation = 0  # Current generation
        # All time best evaluations, ordered from best to worst fitness
        self.best_list = SortedListWithKey(key=lambda x: -x.fitness)
        self.best_set = set()
        self.best_test = None  # GenomeEvaluation (evaluated with the test data-set) of the best individual in best_test

    def clear(self):
        self.initial_time = None
        self.eval_time = datetime.timedelta()
        self.ea_time = datetime.timedelta()
        self.cur_run = None
        self.cur_window = None
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
            g = neat.Genome(0, self.data.get_num_inputs(), 0, 1, False, output_act_f, hidden_act_f, 0, self.params, 0)
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

    def is_finished(self):
        generation_limit = self.options.generations is not None and self.generation >= self.options.generations
        time_limit = self.options.time_limit is not None and \
                     self.elapsed_time().total_seconds() / 60 >= self.options.time_limit
        return generation_limit or time_limit

    def print(self, msg, override=False):
        if not self.options.quiet or override:
            print(msg)

    def make_out_dir(self):
        if self.options.out_dir is None:
            raise ValueError('out_dir is None')

        if not os.path.exists(self.options.out_dir):
            os.makedirs(self.options.out_dir)

    def get_out_file_path(self, suffix):
        if self.options.out_dir is None:
            raise ValueError('out_dir is None')

        run = '({})'.format(self.cur_run) if self.cur_run is not None else ''

        # Format: '<OUTDIR>/<METHOD>_<ID><(RUN_INDEX)>_<SUFFIX>
        # Example: 'results/neat_sample1K(0)_summary.json'
        return '{}/{}_{}{}_{}'.format(self.options.out_dir, self.options.method, self.options.id, run, suffix)

    def write_results(self):
        if self.options.out_dir is not None:
            self.make_out_dir()
            self.write_summary(self.get_out_file_path('summary.json'))
            self.pop.Save(self.get_out_file_path('population.txt'))
            self.get_best().genome.Save(self.get_out_file_path('best.txt'))

    def get_summary(self):
        best_evaluation = self.get_best()
        net = util.build_network(best_evaluation.genome, self.options.method, self.substrate)
        return Summary(best_eval=best_evaluation, best_network=net,
                       fitness_test=self.best_test.fitness if self.best_test is not None else None,
                       # Other info
                       params=params.ParametersWrapper(self.params), generations=self.generation,
                       run_time=datetime.datetime.now() - self.initial_time, eval_time=self.eval_time,
                       ea_time=self.ea_time, processes=self.options.processes,
                       sample_size=self.options.sample_size if self.options.sample_size is not None \
                           else len(self.data))

    def write_summary(self, file_path):
        with open(file_path, 'w', encoding='utf8') as f:
            f.write(json.dumps(self.get_summary().__dict__, default=util.serializer, indent=4))

    def save_evaluations(self, evaluations):
        if self.options.no_statistics or self.options.out_dir is None:
            return

        self.make_out_dir()
        file_path = self.get_out_file_path('evaluations.csv')
        if self.generation == 0:
            with open(file_path, 'w') as file:
                writer = csv.writer(file, delimiter=',')
                header = ['generation', 'genome_id', 'fitness', 'fitness_test', 'neurons', 'connections', 'run_minutes']
                writer.writerow(header)
        with open(file_path, 'a') as file:
            writer = csv.writer(file, delimiter=',')
            for e in evaluations:
                fitness_test = e.fitness_test if e.fitness_test is not None else -1
                writer.writerow([e.generation, e.genome_id, e.fitness, fitness_test, e.neurons, e.connections,
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
                self._add_to_best_list(e)

                self.print("New best (#{})> Fitness: {:.6f}, Neurons: {}, Connections:{}".
                           format(i, e.fitness, e.neurons, e.connections))

                # Cap the size of best_list at PopulationSize
                if len(self.best_list) > self.params.PopulationSize:
                    self._remove_from_best_list(-1)

    def _add_to_best_list(self, evaluation):
        evaluation.save_genome_copy()
        i = self.best_list.bisect_left(evaluation)
        self.best_list.insert(i, evaluation)
        self.best_set.add(evaluation.genome_id)

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
            genome_list,
            partial(self.genome_evaluator, method=self.options.method,
                    substrate=self.substrate, generation=self.generation, initial_time=self.initial_time),
            data=self.data, sample_size=sample_size, processes=self.options.processes, test_data=test_data
        )

    def evaluate(self, genome):
        return self._evaluate(genome, self.data)

    def evaluate_test(self, genome):
        return self._evaluate(genome, self.test_data)

    def _evaluate(self, genome, data):
        return self.genome_evaluator(genome, data, generation=self.generation, initial_time=self.initial_time)

    def evaluate_pop(self):
        pre_eval_time = datetime.datetime.now()
        evaluation_list = self.evaluate_list(self.get_genome_list())
        self.eval_time += datetime.datetime.now() - pre_eval_time

        self.save_evaluations(evaluation_list)
        self.update_best_list(evaluation_list)

    def epoch(self):
        pre_ea_time = datetime.datetime.now()
        self.pop.Epoch()
        self.ea_time += datetime.datetime.now() - pre_ea_time
        self.generation += 1

    def print_best(self):
        best = self.get_best()
        best_test_str = ' Fitness (test): {:.6f},'.format(self.best_test.fitness) if self.best_test is not None else ''

        self.print("\nBest result> Fitness (train): {:.6f},{} Neurons: {}, Connections: {}".
                   format(best.fitness, best_test_str, best.neurons, best.connections), override=True)

    def evaluate_best_test(self):
        best = self.get_best()
        if self.options.test_file is not None:
            self.best_test = self.evaluate_test(best.genome)

    def _run(self):
        self.initial_time = datetime.datetime.now()

        # Run the EA
        while not self.is_finished():
            self.print("\nGeneration {} ({})".format(self.generation, self.elapsed_time()))
            self.evaluate_pop()
            self.epoch()

        # Reevaluate the best individuals with full data if sample_size is specified
        if self.options.sample_size != 0 and not self.options.no_reevaluation:
            self.print("\nReevaluating the best individuals with the whole data-set...")
            self.reevaluate_best_list()

        self.evaluate_best_test()  # Test the best individual obtained with the test data-set
        self.print_best()  # Print to stdout the best result
        self.write_results()  # Write run details to files

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
            self.cur_run = i
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
    # evolver = Evolver(options)
    # evolver.run()
    # FIXME wip...
    import data
    data.main(options)