#!/usr/bin/python3

import MultiNEAT as neat
import os
import time
import argparse

from params import get_params, ParametersWrapper
import evaluators
import util
import substrate as subst

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
    parser.add_argument('-g', '--generations', dest='generations', type=int, default=250,
                        help='number of generations', metavar='G')
    parser.add_argument('-p', '--processes', dest='processes', type=int, default=1,
                        help='number of processes to use for parallel computation. '
                             'If P=1, the execution will be sequential', metavar='P')
    parser.add_argument('-P', '--population', dest='pop_file', metavar='FILE', default=None,
                        help='load the contents of FILE as the initial population and parameters')
    parser.add_argument('-i', '--id', dest='run_id', metavar='ID', default=None,
                        help='run identifier. This ID will be used to name all output files '
                             '(e.g., neat_ID_summary.txt). '
                             'If unspecified, the ID will be the datetime of when the run terminates.')

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


def init_population(params, options, seed=int(time.clock() * 100), substrate=None):
    if options.pop_file is not None:
        return neat.Population(options.pop_file)

    pop = None
    output_act_f = neat.ActivationFunction.UNSIGNED_SIGMOID
    hidden_act_f = neat.ActivationFunction.UNSIGNED_SIGMOID

    if options.method == 'neat':
        g = neat.Genome(0, 10 + 1, 0, 1, False, output_act_f, hidden_act_f, 0, params, 0)
        pop = neat.Population(g, params, True, 1.0, 0)
    elif options.method in ['hyperneat', 'eshyperneat']:
        g = neat.Genome(0, substrate.GetMinCPPNInputs(), 0, substrate.GetMinCPPNOutputs(),
                        False, output_act_f, hidden_act_f, 0, params, 0)
        pop = neat.Population(g, params, True, 1.0, 0)

    pop.RNG.Seed(seed)
    return pop


def main():
    options, list_evaluator, genome_evaluator = parse_args()
    params = get_params()
    substrate = subst.init_2d_grid_substrate(11, 10, [10] * 10, 1) if \
        options.method in ['hyperneat', 'eshyperneat'] else None

    initial_time = datetime.datetime.now()
    eval_time = datetime.timedelta()
    ea_time = datetime.timedelta()

    data = util.read_data(options.data_file)
    true_targets = np.array([row['target'] for row in data])

    gen_evaluations = dict()  # Dict<Generation, [GenomeEvaluation]>
    all_time_best = None

    pop = init_population(params, options, substrate=substrate)
    params = pop.Parameters  # Needed in case pop was loaded from a file

    for generation in range(options.generations):
        print("Generation {}...".format(generation))
        genome_list = util.get_genome_list(pop)

        pre_eval_time = datetime.datetime.now()
        evaluation_list = list_evaluator(
            genome_list, partial(genome_evaluator, data=data, true_targets=true_targets,
                                 processes=options.processes, method=options.method, substrate=substrate)
        )
        eval_time += datetime.datetime.now() - pre_eval_time
        gen_evaluations[generation] = evaluation_list

        best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
        print("[DEBUG] Best fitness of generation {}: {}".format(generation, best_evaluation.fitness))

        if all_time_best is None or best_evaluation.fitness > all_time_best.fitness:
            all_time_best = best_evaluation
            all_time_best.save_genome_copy()
            print_info(all_time_best, options.method, substrate)

        pre_ea_time = datetime.datetime.now()
        pop.Epoch()
        ea_time += datetime.datetime.now() - pre_ea_time

    file_prefix = '{}/{}_{}_'.format(options.out_dir, options.method, util.get_current_datetime_string())
    util.write_summary(out_file_path=file_prefix + 'summary.json', best_evaluation=all_time_best,
                       params=ParametersWrapper(params), generations=options.generations,
                       run_time=datetime.datetime.now() - initial_time, eval_time=eval_time, ea_time=ea_time,
                       evaluation_processes=options.processes)
    util.save_evaluations(file_prefix + 'evaluations.csv', gen_evaluations)
    pop.Save(file_prefix + 'population.txt')
    all_time_best.genome.Save(file_prefix + 'best.txt')


if __name__ == '__main__':
    main()
