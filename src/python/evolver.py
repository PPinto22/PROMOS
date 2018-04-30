#!/usr/bin/python3

import MultiNEAT as neat
import os
import time
import argparse

from params import get_params, ParametersWrapper
import evaluators
import util

import datetime
from functools import partial
import numpy as np


def parse_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', dest='data_file', help='path to input data file', metavar='FILE',
                        default='../../data/data.csv')
    parser.add_argument('-o', '--outdir', dest='out_dir', help='directory where to save results', metavar='DIR',
                        default='../../results')
    methods = ['neat', 'hyperneat', 'eshyperneat']
    parser.add_argument('-m', '--method', dest='method', help='which algorithm to run: ' + ', '.join(methods),
                        metavar='M', choices=methods, default='neat')
    evaluation_functions = ['auc']
    parser.add_argument('-e', '--evaluator', dest='evaluator',
                        help='evaluation function: ' + ', '.join(evaluation_functions),
                        metavar='E', choices=evaluation_functions, default='auc')
    parser.add_argument('-g', '--generations', dest='generations', help='number of generations', metavar='G',
                        type=int, default=150)
    parser.add_argument('-p', '--processes', dest='processes',
                        help='number of processes to use for parallel computation. '
                             'If P=1, the execution will be sequential', metavar='P',
                        type=int, default=1)

    options = parser.parse_args()
    list_evaluator = evaluators.evaluate_genome_list_serial if options.processes == 1 else \
        evaluators.evaluate_genome_list_parallel
    if options.evaluator == 'auc':
        genome_evaluator = evaluators.evaluate_auc
    else:
        raise ValueError('Invalid genome evaluator: {}'.format(options.evaluator))

    return options, list_evaluator, genome_evaluator


def print_info(evaluation, method, substrate):
    network = util.build_network(evaluation.genome, method, substrate)
    print("[DEBUG] New Best!")
    print("[DEBUG] Fitness: " + str(evaluation.fitness))
    print("[DEBUG] Neurons: " + str(len(util.get_network_neurons(network))))
    print("[DEBUG] Connections: " + str(len(util.get_network_connections(network))))


def init_substrate():
    substrate = neat.Substrate(
        # Input
        [(-1, -1), (-1, -0.8), (-1, -0.6), (-1, -0.4), (-1, -0.2),
         (-1, 0.0), (-1, 0.2), (-1, 0.4), (-1, 0.6), (-1, 0.8), (-1, 1)],  # (-1, 1) = Bias
        # Hidden
        [],
        # Output
        [(1, 0)]
    )
    return substrate


def init_population(params, method='neat', seed=int(time.clock() * 100), substrate=None):
    pop = None
    output_act_f = neat.ActivationFunction.UNSIGNED_SIGMOID
    hidden_act_f = neat.ActivationFunction.UNSIGNED_SIGMOID

    if method == 'neat':
        g = neat.Genome(0, 10 + 1, 0, 1, False, output_act_f, hidden_act_f, 0, params, 0)
        pop = neat.Population(g, params, True, 1.0, 0)
    elif method in ['hyperneat', 'eshyperneat']:
        print(substrate.GetMinCPPNInputs())
        print(substrate.GetMinCPPNOutputs())
        g = neat.Genome(0, substrate.GetMinCPPNInputs(), 0, substrate.GetMinCPPNOutputs(),
                        False, output_act_f, hidden_act_f, 0, params, 0)
        pop = neat.Population(g, params, True, 1.0, 0)

    pop.RNG.Seed(seed)
    return pop


def main():
    options, list_evaluator, genome_evaluator = parse_options()
    params = get_params()
    substrate = init_substrate()

    initial_time = datetime.datetime.now()
    eval_time = datetime.timedelta()
    ea_time = datetime.timedelta()

    data = util.read_data(options.data_file)
    true_targets = np.array([row['target'] for row in data])

    pop = init_population(params, method=options.method, substrate=substrate)

    all_time_best = None
    for generation in range(options.generations):
        print("Generation {}...".format(generation))
        genome_list = util.get_genome_list(pop)

        pre_eval_time = datetime.datetime.now()
        evaluation_list = list_evaluator(
            genome_list, partial(genome_evaluator, data=data, true_targets=true_targets,
                                 processes=options.processes, method=options.method, substrate=substrate)
        )
        eval_time += datetime.datetime.now() - pre_eval_time

        best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
        print("[DEBUG] Best fitness of generation {}: {}".format(generation, best_evaluation.fitness))

        if all_time_best is None or best_evaluation.fitness > all_time_best.fitness:
            all_time_best = best_evaluation
            all_time_best.save_genome_copy()
            print_info(all_time_best, options.method, substrate)

        pre_ea_time = datetime.datetime.now()
        pop.Epoch()
        ea_time += datetime.datetime.now() - pre_ea_time

    date_time = util.get_current_datetime_string()
    util.write_results(
        out_file_path='{}/{}_{}_summary.json'.format(options.out_dir, options.method, date_time),
        best_evaluation=all_time_best,
        # -- Other info --
        params=ParametersWrapper(params),
        generations=options.generations,
        run_time=datetime.datetime.now() - initial_time,
        eval_time=eval_time,
        ea_time=ea_time,
        evaluation_processes=options.processes
    )
    pop.Save('{}/{}_{}_population.txt'.format(options.out_dir, options.method, date_time))
    all_time_best.genome.Save('{}/{}_{}_best.txt'.format(options.out_dir, options.method, date_time))


if __name__ == '__main__':
    main()
