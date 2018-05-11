#!/usr/bin/python3

import MultiNEAT as neat
import os
import time
import argparse

from params import get_params, ParametersWrapper
import evaluators
import util
import substrate as subst
from data import Data

import datetime
from functools import partial
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', dest='data_file', default='../../data/data.csv',
                        help='path to input data file', metavar='FILE')
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
                        help='number of processes to use for parallel computation. '
                             'If P=1, the execution will be sequential', metavar='P')
    parser.add_argument('-s', '--sample', dest='sample_size', type=int, default=None,
                        help='use a balanced sample of size N in evaluations', metavar='N')
    parser.add_argument('-P', '--population', dest='pop_file', metavar='FILE', default=None,
                        help='load the contents of FILE as the initial population and parameters')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='Do not print any messages to stdout')
    parser.add_argument('--id', dest='run_id', metavar='ID', default=None,
                        help='run identifier. This ID will be used to name all output files '
                             '(e.g., neat_ID_summary.txt). '
                             'If unspecified, the ID will be the datetime of when the run was started.')

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

        # MultiNEAT parameters
        self.params = get_params()
        # Substrate for HyperNEAT and ES-HyperNEAT
        self.substrate = subst.init_2d_grid_substrate(11, 10, [10] * 10, 1) if \
            options.method in ['hyperneat', 'eshyperneat'] else None

        # Data
        self.data = Data(self.options.data_file)

        # Timers
        self.initial_time = None  # Time when the run starts
        self.eval_time = datetime.timedelta()  # Time spent in evaluations
        self.ea_time = datetime.timedelta()  # Time spent in the EA

        # Progress variables
        self.gen_evaluations = dict()  # Dict<Generation, [GenomeEvaluation]>
        self.all_time_best = None  # evaluators.GenomeEvaluation

        # EA variables
        self.pop = self.init_population()  # C++ Population
        self.params = self.pop.Parameters  # Needed in case pop is loaded from file
        self.generation = 0  # Current generation

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

    def update_best(self, new_best_eval):
        new_best_eval.save_genome_copy()
        self.all_time_best = new_best_eval
        if not self.options.quiet:
            network = util.build_network(new_best_eval.genome, self.options.method, self.substrate)
            print("New Best!")
            print("Fitness: " + str(new_best_eval.fitness))
            print("Neurons: " + str(len(util.get_network_neurons(network))))
            print("Connections: " + str(len(util.get_network_connections(network))))

    def print(self, msg):
        if not self.options.quiet:
            print(msg)

    def write_results(self):
        file_prefix = '{}/{}_{}_'.format(self.options.out_dir, self.options.method, self.options.run_id)
        if not os.path.exists(self.options.out_dir):
            os.makedirs(self.options.out_dir)
        util.write_summary(file_prefix + 'summary.json', self.all_time_best, self.options.method, self.substrate,
                           params=ParametersWrapper(self.params), generations=self.options.generations,
                           run_time=datetime.datetime.now() - self.initial_time, eval_time=self.eval_time,
                           ea_time=self.ea_time, evaluation_processes=self.options.processes,
                           sample_size=self.options.sample_size if self.options.sample_size is not None \
                               else len(self.data))
        util.save_evaluations(file_prefix + 'evaluations.csv', self.gen_evaluations)
        self.pop.Save(file_prefix + 'population.txt')
        self.all_time_best.genome.Save(file_prefix + 'best.txt')

    def run(self):
        self.initial_time = datetime.datetime.now()
        while not self.is_finished():
            self.print("\nGeneration {} ({})".format(self.generation, self.elapsed_time()))
            genome_list = self.get_genome_list()

            pre_eval_time = datetime.datetime.now()
            evaluation_list = evaluators.evaluate_genome_list(
                genome_list,
                partial(self.genome_evaluator, method=options.method,
                        substrate=self.substrate, initial_time=self.initial_time),
                self.data, options.sample_size, options.processes
            )
            self.eval_time += datetime.datetime.now() - pre_eval_time
            self.gen_evaluations[self.generation] = evaluation_list

            best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
            self.print("Best fitness of generation {}: {}".format(self.generation, best_evaluation.fitness))
            if self.all_time_best is None or best_evaluation.fitness > self.all_time_best.fitness:
                # FIXME This doesn't make sense when there's sampling
                self.update_best(best_evaluation)

            pre_ea_time = datetime.datetime.now()
            self.pop.Epoch()
            self.ea_time += datetime.datetime.now() - pre_ea_time

            self.generation += 1
        self.write_results()
        # FIXME Random segmentation fault at the end


if __name__ == '__main__':
    options = parse_args()
    evolver = Evolver(options)
    evolver.run()
