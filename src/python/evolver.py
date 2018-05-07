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
    parser.add_argument('-g', '--generations', dest='generations', type=int, default=250,
                        help='number of generations', metavar='G')
    parser.add_argument('-p', '--processes', dest='processes', type=int, default=1,
                        help='number of processes to use for parallel computation. '
                             'If P=1, the execution will be sequential', metavar='P')
    parser.add_argument('-s', '--sample', dest='sample_size', type=int, default=None,
                        help='use a balanced sample of size N in evaluations', metavar='N')
    parser.add_argument('-P', '--population', dest='pop_file', metavar='FILE', default=None,
                        help='load the contents of FILE as the initial population and parameters')
    parser.add_argument('-i', '--id', dest='run_id', metavar='ID', default=None,
                        help='run identifier. This ID will be used to name all output files '
                             '(e.g., neat_ID_summary.txt). '
                             'If unspecified, the ID will be the datetime of when the run was started.')

    options = parser.parse_args()

    options.run_id = options.run_id if options.run_id is not None else util.get_current_datetime_string()

    if options.evaluator == 'auc':
        genome_evaluator = evaluators.evaluate_auc
    else:
        raise ValueError('Invalid genome evaluator: {}'.format(options.evaluator))

    return options, genome_evaluator


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
    options, genome_evaluator = parse_args()
    params = get_params()
    substrate = subst.init_2d_grid_substrate(11, 10, [10] * 10, 1) if \
        options.method in ['hyperneat', 'eshyperneat'] else None

    initial_time = datetime.datetime.now()
    eval_time = datetime.timedelta()
    ea_time = datetime.timedelta()

    data = Data(options.data_file)
    gen_evaluations = dict()  # Dict<Generation, [GenomeEvaluation]>
    all_time_best = None

    pop = init_population(params, options, substrate=substrate)
    params = pop.Parameters  # Needed in case pop was loaded from a file

    for generation in range(options.generations):
        print("Generation {}...".format(generation))
        genome_list = util.get_genome_list(pop)

        pre_eval_time = datetime.datetime.now()
        evaluation_list = evaluators.evaluate_genome_list(
            genome_list,
            partial(genome_evaluator, method=options.method, substrate=substrate, initial_time=initial_time),
            data, options.sample_size, options.processes
        )
        eval_time += datetime.datetime.now() - pre_eval_time
        gen_evaluations[generation] = evaluation_list

        best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
        print("[DEBUG] Best fitness of generation {}: {}".format(generation, best_evaluation.fitness))

        if all_time_best is None or best_evaluation.fitness > all_time_best.fitness:
            # TODO Faz sentido guardar o melhor desta forma quando há sampling???
            all_time_best = best_evaluation
            all_time_best.save_genome_copy()
            print_info(all_time_best, options.method, substrate)

        pre_ea_time = datetime.datetime.now()
        pop.Epoch()
        ea_time += datetime.datetime.now() - pre_ea_time

    file_prefix = '{}/{}_{}_'.format(options.out_dir, options.method, options.run_id)
    if not os.path.exists(options.out_dir):
        os.makedirs(options.out_dir)
    util.write_summary(file_prefix + 'summary.json', all_time_best, options.method, substrate,
                       params=ParametersWrapper(params), generations=options.generations,
                       run_time=datetime.datetime.now() - initial_time, eval_time=eval_time, ea_time=ea_time,
                       evaluation_processes=options.processes,
                       sample_size=options.sample_size if options.sample_size is not None else len(data))
    util.save_evaluations(file_prefix + 'evaluations.csv', gen_evaluations)
    pop.Save(file_prefix + 'population.txt')
    all_time_best.genome.Save(file_prefix + 'best.txt')

    # FIXME Random segmentation fault at the end


if __name__ == '__main__':
    main()
