#!/usr/bin/python3

import MultiNEAT as neat
import os

from params import get_params, ParametersWrapper
import evaluators
import util

import datetime
from functools import partial
import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# from viz import Draw

DATA_FILE_PATH = '../../data/data_micro.csv'
OUT_DIR = '../../results'

GENERATIONS = 250
PARAMS = get_params()

LIST_EVALUATOR = evaluators.evaluate_genome_list_parallel
EVALUATION_PROCESSES = 54 #os.cpu_count() or 1
GENOME_EVALUATOR = evaluators.evaluate_auc

if __name__ == '__main__':
    initial_time = datetime.datetime.now()
    eval_time = datetime.timedelta()
    ea_time = datetime.timedelta()

    data = util.read_data(DATA_FILE_PATH)
    true_targets = np.array([row['target'] for row in data])

    g = neat.Genome(0, 10, 0, 1, False, neat.ActivationFunction.UNSIGNED_SIGMOID,
                    neat.ActivationFunction.UNSIGNED_SIGMOID, 0, PARAMS, 5)
    pop = neat.Population(g, PARAMS, True, 1.0, 0)  # 0 is the RNG seed
    # pop.RNG.Seed(int(time.clock() * 100))

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

        # Plot network
        # cv2.imshow("Best Network", Draw(best_evaluation.network))
        # cv2.waitKey(1)

        # Plot ROC
        # roc = best_evaluation.metrics.roc
        # fig = plt.gcf()
        # fig.canvas.set_window_title('ROC Curve')
        # plt.clf()
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.005])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.plot(roc.fpr, roc.tpr, color='darkorange',
        #          lw=2, label='ROC Curve (area = %.2f)' % roc.auc)
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # plt.title('Generation {}\'s Best Network'.format(generation))
        # plt.legend(loc="lower right")
        # plt.draw()
        # plt.pause(0.001)

        pre_ea_time = datetime.datetime.now()
        pop.Epoch()
        ea_time += datetime.datetime.now() - pre_ea_time

    util.write_results(
        out_file_path='{}/neat_{}.json'.format(OUT_DIR, util.get_current_datetime_string()),
        best_evaluation=all_time_best,
        # -- Other info --
        params=ParametersWrapper(PARAMS),
        generations=GENERATIONS,
        run_time=datetime.datetime.now() - initial_time,
        eval_time=eval_time,
        ea_time=ea_time,
        evaluation_processes=EVALUATION_PROCESSES if LIST_EVALUATOR is evaluators.evaluate_genome_list_parallel else 1
    )
