#!/usr/bin/python3

import MultiNEAT as neat
import csv
import json
import math
import os
import time
import argparse
import datetime
from sortedcontainers import SortedListWithKey
from functools import partial

from params import get_params, ParametersWrapper
import evaluators
import util
import substrate as subst
from data import Data


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', dest='data_file', default='../../data/data.csv',
                        help='path to train data file', metavar='FILE'),
    parser.add_argument('-T', '--test', dest='test_file', default=None,
                        help='path to test data file', metavar='FILE')
    parser.add_argument('-o', '--outdir', dest='out_dir', default='../../results',
                        help='directory where to save results', metavar='DIR')
    methods = ['neat', 'hyperneat', 'eshyperneat']
    parser.add_argument('-m', '--method', dest='method', choices=methods, default='neat',
                        help='which algorithm to run: ' + ', '.join(methods), metavar='M')
    evaluation_functions = ['auc']
    parser.add_argument('-e', '--evaluator', dest='evaluator', choices=evaluation_functions, default='auc',
                        help='evaluation function: ' + ', '.join(evaluation_functions), metavar='E')
    parser.add_argument('-g', '--generations', dest='generations', type=int, default=None,
                        help='number of generations', metavar='G')
    parser.add_argument('-t', '--time', dest='time_limit', type=int, default=None,
                        help='time limit in minutes', metavar='MIN')
    parser.add_argument('-p', '--processes', dest='processes', type=int, default=1,
                        help='number of processes to use for parallel evaluation. '
                             'If P=1, the evaluations will be sequential', metavar='P')
    parser.add_argument('-s', '--sample', dest='sample_size', type=int, default=None,
                        help='use a balanced sample of size N in evaluations', metavar='N')
    parser.add_argument('-P', '--population', dest='pop_file', metavar='FILE', default=None,
                        help='load the contents of FILE as the initial population and parameters')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='Do not print any messages to stdout')
    parser.add_argument('--id', dest='run_id', metavar='ID', default=None,
                        help='run identifier. This ID will be used to name all output files '
                             '(e.g., neat_ID_summary.txt). '
                             'If unspecified, the ID will be the datetime of when the run was started.')
    parser.add_argument('--no-reevaluation', dest='no_reevaluation', action='store_true',
                        help='applicable if a sample size is specified. '
                             'If set, there will be no final reevaluation of '
                             'the best individuals with the whole data-set.')

    options = parser.parse_args()

    options.run_id = options.run_id if options.run_id is not None else util.get_current_datetime_string()

    return options


class Evolver:
    def __init__(self, options):
        self.options = options

        # Evaluation function
        if options.evaluator == 'auc':
            self.genome_evaluator = evaluators.evaluate_auc
        else:
            raise ValueError('Invalid genome evaluator: {}'.format(options.evaluator))

        self.params = get_params()  # MultiNEAT parameters
        # Substrate for HyperNEAT and ES-HyperNEAT
        self.substrate = subst.init_2d_grid_substrate(11, 10, [10] * 10, 1) if \
            options.method in ['hyperneat', 'eshyperneat'] else None

        self.data = Data(self.options.data_file)  # Training data
        self.data_test = Data(self.options.test_file) if self.options.test_file is not None else None  # Test data

        self.initial_time = None  # Time when the run starts
        self.eval_time = datetime.timedelta()  # Time spent in evaluations
        self.ea_time = datetime.timedelta()  # Time spent in the EA

        self.pop = self.init_population()  # C++ Population
        self.params = self.pop.Parameters  # Needed in case pop is loaded from file
        self.generation = 0  # Current generation
        # All time best evaluations, ordered from best to worst fitness
        self.best_list = SortedListWithKey(key=lambda x: -x.fitness)
        self.best_test = None  # GenomeEvaluation of the best individual, evaluated with the test data-set

    def init_population(self, seed=int(time.clock() * 100)):
        if self.options.pop_file is not None:
            return neat.Population(self.options.pop_file)

        pop = None
        output_act_f = neat.ActivationFunction.UNSIGNED_SIGMOID
        hidden_act_f = neat.ActivationFunction.UNSIGNED_SIGMOID

        if self.options.method == 'neat':
            g = neat.Genome(0, 10, 0, 1, False, output_act_f, hidden_act_f, 0, self.params, 0)
            pop = neat.Population(g, self.params, True, 1.0, 0)
        elif self.options.method in ['hyperneat', 'eshyperneat']:
            g = neat.Genome(0, self.substrate.GetMinCPPNInputs(), 0, self.substrate.GetMinCPPNOutputs(),
                            False, output_act_f, hidden_act_f, 0, self.params, 0)
            pop = neat.Population(g, self.params, True, 1.0, 0)

        pop.RNG.Seed(seed)
        return pop

    def get_genome_list(self):
        genome_list = []
        for s in self.pop.Species:
            for i in s.Individuals:
                genome_list.append(i)
        return genome_list

    def elapsed_time(self):
        return datetime.datetime.now() - self.initial_time

    def is_finished(self):
        generation_limit = self.options.generations is not None and self.generation >= self.options.generations
        time_limit = self.options.time_limit is not None and \
                     self.elapsed_time().total_seconds() / 60 >= self.options.time_limit
        return generation_limit or time_limit

    def print(self, msg):
        if not self.options.quiet:
            print(msg)

    def make_out_dir(self):
        if not os.path.exists(self.options.out_dir):
            os.makedirs(self.options.out_dir)

    def get_out_file_path(self, suffix):
        return '{}/{}_{}_{}'.format(self.options.out_dir, self.options.method, self.options.run_id, suffix)

    def write_results(self):
        self.make_out_dir()
        self.write_summary(self.get_out_file_path('summary.json'))
        self.pop.Save(self.get_out_file_path('population.txt'))
        self.get_best().genome.Save(self.get_out_file_path('best.txt'))

    def write_summary(self, file_path):
        class Summary:
            class Network:
                def __init__(self, eval, test, network=None):
                    self.fitness = eval.fitness
                    self.fitness_test = test
                    if network is not None:
                        self.connections = util.get_network_connections(network)
                        self.neurons = util.get_network_neurons(network)
                        self.neurons_qty = len(self.neurons)
                        self.connections_qty = len(self.connections)

            def __init__(self, best_eval, fitness_test, best_network=None, **other_info):
                self.best = Summary.Network(best_eval, fitness_test, best_network)
                for key, value in other_info.items():
                    self.__setattr__(key, value)

        best_evaluation = self.get_best()
        net = util.build_network(best_evaluation.genome, self.options.method, self.substrate)
        results = Summary(best_eval=best_evaluation, best_network=net,
                          fitness_test=self.best_test.fitness if self.best_test is not None else None,
                          # Other info
                          params=ParametersWrapper(self.params),
                          generations=self.options.generations,
                          run_time=datetime.datetime.now() - self.initial_time, eval_time=self.eval_time,
                          ea_time=self.ea_time, evaluation_processes=self.options.processes,
                          sample_size=self.options.sample_size if self.options.sample_size is not None \
                              else len(self.data))
        with open(file_path, 'w', encoding='utf8') as f:
            f.write(json.dumps(results.__dict__, default=util.serializer, indent=4))

    def save_evaluations(self, evaluations):
        self.make_out_dir()
        file_path = self.get_out_file_path('evaluations.csv')
        if self.generation == 0:
            with open(file_path, 'w') as file:
                writer = csv.writer(file, delimiter=',')
                header = ['generation', 'genome_id', 'fitness', 'neurons', 'connections', 'run_minutes']
                writer.writerow(header)
        with open(file_path, 'a') as file:
            writer = csv.writer(file, delimiter=',')
            for e in evaluations:
                writer.writerow([e.generation, e.genome_id, e.fitness, e.neurons, e.connections,
                                 e.global_time.total_seconds() / 60.0])

    def get_best(self):
        return self.best_list[0]

    def update_best_list(self, evaluations):
        updated = 0
        max_updates = math.ceil(0.1 * self.params.PopulationSize)  # Take at most 10% of evaluations
        # Evaluations should be sorted by descending fitness
        for e in evaluations:
            if len(self.best_list) <= self.params.PopulationSize or e.fitness >= self.best_list[-1].fitness:
                i = self.best_list.bisect_left(e)
                if i < self.params.PopulationSize:
                    e.save_genome_copy()
                    self.best_list.insert(i, e)

                    self.print("New best (#{})> Fitness: {:.6f}, Neurons: {}, Connections:{}".
                               format(i, e.fitness, e.neurons, e.connections))

                    if len(self.best_list) > self.params.PopulationSize:
                        del self.best_list[-1]

                    updated += 1
                    if updated >= max_updates:
                        break
                else:
                    break

    def reevaluate_best_list(self):
        genome_list = [e.genome for e in self.best_list]
        evaluation_list = evaluators.evaluate_genome_list(
            genome_list,
            partial(self.genome_evaluator, method=options.method,
                    substrate=self.substrate, generation=self.generation, initial_time=self.initial_time),
            data=self.data, sample_size=None, processes=options.processes
        )
        self.best_list.clear()
        for e in evaluation_list:
            self.best_list.add(e)

    def evaluate_list(self, genome_list):
        return evaluators.evaluate_genome_list(
            genome_list,
            partial(self.genome_evaluator, method=options.method,
                    substrate=self.substrate, generation=self.generation, initial_time=self.initial_time),
            self.data, options.sample_size, options.processes
        )

    def evaluate(self, genome):
        return self._evaluate(genome, self.data)

    def evaluate_test(self, genome):
        return self._evaluate(genome, self.data_test)

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
                   format(best.fitness, best_test_str, best.neurons, best.connections))

    def evaluate_best_test(self):
        best = self.get_best()
        if self.options.test_file is not None:
            self.best_test = self.evaluate_test(best.genome)

    def run(self):
        self.initial_time = datetime.datetime.now()

        # Run the EA
        while not self.is_finished():
            self.print("\nGeneration {} ({})".format(self.generation, self.elapsed_time()))
            self.evaluate_pop()
            self.epoch()

        # Reevaluate the best individuals with full data if sample_size is specified
        if self.options.sample_size is not None and not self.options.no_reevaluation:
            self.print("\nReevaluating the best individuals with the whole data-set...")
            self.reevaluate_best_list()

        self.evaluate_best_test()  # Test the best individual obtained with the test data-set
        self.print_best()  # Print to stdout the best result
        self.write_results()  # Write run details to files


if __name__ == '__main__':
    options = parse_args()
    evolver = Evolver(options)
    evolver.run()
