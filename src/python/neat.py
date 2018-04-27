#!/usr/bin/python3

import MultiNEAT as neat
import os
import time

from params import get_params, ParametersWrapper
import evaluators
import util

import datetime
from functools import partial
import numpy as np


DATA_FILE_PATH = '../../data/data.csv'
OUT_DIR = '../../results'

GENERATIONS = 250
PARAMS = get_params()

LIST_EVALUATOR = evaluators.evaluate_genome_list_parallel
EVALUATION_PROCESSES = os.cpu_count() or 1
GENOME_EVALUATOR = evaluators.evaluate_auc


def print_info(evaluation):
    network = util.build_network(evaluation.genome)
    print("[DEBUG] New Best!")
    print("[DEBUG] Fitness: " + str(evaluation.fitness))
    print("[DEBUG] Neurons: " + str(len(util.get_network_neurons(network))))
    print("[DEBUG] Connections: " + str(len(util.get_network_connections(network))))


if __name__ == '__main__':
    initial_time = datetime.datetime.now()
    eval_time = datetime.timedelta()
    ea_time = datetime.timedelta()

    data = util.read_data(DATA_FILE_PATH)
    true_targets = np.array([row['target'] for row in data])

    g = neat.Genome(0, 10+1, 0, 1, False, neat.ActivationFunction.UNSIGNED_SIGMOID,
                    neat.ActivationFunction.UNSIGNED_SIGMOID, 0, PARAMS, 5)
    pop = neat.Population(g, PARAMS, True, 1.0, 0)  # 0 is the RNG seed
    pop.RNG.Seed(int(time.clock() * 100))

    all_time_best = None
    for generation in range(GENERATIONS):
        print("Generation {}...".format(generation))
        genome_list = util.get_genome_list(pop)

        pre_eval_time = datetime.datetime.now()
        evaluation_list = LIST_EVALUATOR(
            genome_list, partial(GENOME_EVALUATOR, data=data, true_targets=true_targets, processes=EVALUATION_PROCESSES)
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
        out_file_path='{}/neat_{}_summary.json'.format(OUT_DIR, date_time),
        best_evaluation=all_time_best,
        # -- Other info --
        params=ParametersWrapper(PARAMS),
        generations=GENERATIONS,
        run_time=datetime.datetime.now() - initial_time,
        eval_time=eval_time,
        ea_time=ea_time,
        evaluation_processes=EVALUATION_PROCESSES if LIST_EVALUATOR is evaluators.evaluate_genome_list_parallel else 1
    )
    pop.Save('{}/neat_{}_population.txt'.format(OUT_DIR, date_time))
    all_time_best.genome.Save('{}/neat_{}_best.txt'.format(OUT_DIR, date_time))
