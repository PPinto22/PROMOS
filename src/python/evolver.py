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

params = get_params()
list_evaluator = evaluators.evaluate_genome_list_serial
genome_evaluator = evaluators.evaluate_auc
options = None


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

    global options
    options = parser.parse_args()

    global list_evaluator
    list_evaluator = evaluators.evaluate_genome_list_serial if options == 1 else \
        evaluators.evaluate_genome_list_parallel

    global genome_evaluator
    if options.evaluator == 'auc':
        genome_evaluator = evaluators.evaluate_auc


def print_info(evaluation):
    network = util.build_network(evaluation.genome)
    print("[DEBUG] New Best!")
    print("[DEBUG] Fitness: " + str(evaluation.fitness))
    print("[DEBUG] Neurons: " + str(len(util.get_network_neurons(network))))
    print("[DEBUG] Connections: " + str(len(util.get_network_connections(network))))


if __name__ == '__main__':
    parse_options()

    initial_time = datetime.datetime.now()
    eval_time = datetime.timedelta()
    ea_time = datetime.timedelta()

    data = util.read_data(options.data_file)
    true_targets = np.array([row['target'] for row in data])

    g = neat.Genome(0, 10 + 1, 0, 1, False, neat.ActivationFunction.UNSIGNED_SIGMOID,
                    neat.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 5)
    pop = neat.Population(g, params, True, 1.0, 0)  # 0 is the RNG seed
    pop.RNG.Seed(int(time.clock() * 100))

    all_time_best = None
    for generation in range(options.generations):
        print("Generation {}...".format(generation))
        genome_list = util.get_genome_list(pop)

        pre_eval_time = datetime.datetime.now()
        evaluation_list = list_evaluator(
            genome_list, partial(genome_evaluator, data=data, true_targets=true_targets, processes=options.processes)
        )
        eval_time += datetime.datetime.now() - pre_eval_time

        best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
        print("[DEBUG] Best fitness of generation {}: {}".format(generation, best_evaluation.fitness))

        if all_time_best is None or best_evaluation.fitness > all_time_best.fitness:
            all_time_best = best_evaluation
            all_time_best.save_genome_copy()
            print_info(all_time_best)

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
    pop.Save('{}/neat_{}_population.txt'.format(options.out_dir, date_time))
    all_time_best.genome.Save('{}/neat_{}_best.txt'.format(options.out_dir, date_time))
