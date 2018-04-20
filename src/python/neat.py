#!/usr/bin/python3

import MultiNEAT as neat

from evaluators import evaluate_genome_list_serial, evaluate_inverse_error, evaluate_auc
from params import get_params
from util import *

# import cv2
# import matplotlib.pyplot as plt

# from viz import Draw

DATA_FILE_PATH = '../../data/data.csv'
OUT_DIR = '../../results'
GENERATIONS = 25

if __name__ == '__main__':
    initial_time = datetime.datetime.now()
    eval_time = datetime.timedelta()
    ea_time = datetime.timedelta()

    data = read_data(DATA_FILE_PATH)
    true_targets = np.array([row['target'] for row in data])
    params = get_params()

    g = neat.Genome(0, 10, 0, 1, False, neat.ActivationFunction.UNSIGNED_SIGMOID,
                    neat.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 5)
    pop = neat.Population(g, params, True, 1.0, 0)  # 0 is the RNG seed
    # pop.RNG.Seed(int(time.clock() * 100))

    all_time_best = None
    for generation in range(GENERATIONS):
        print("--- Generation {} ---".format(generation))
        genome_list = get_genome_list(pop)

        pre_eval_time = datetime.datetime.now()
        evaluation_list = evaluate_genome_list_serial(genome_list,
                                                      lambda genome: evaluate_inverse_error(genome, data, true_targets))
        eval_time += datetime.datetime.now() - pre_eval_time

        best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
        print("[DEBUG] Best fitness of generation {}: {}".format(generation, best_evaluation.fitness))
        if all_time_best is None or best_evaluation.fitness > all_time_best.fitness:
            all_time_best = best_evaluation

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

    write_results('{}/neat_{}.json'.format(OUT_DIR, get_current_datetime_string()), 'neat',
                  GENERATIONS, datetime.datetime.now() - initial_time, all_time_best, params,
                  eval_time=eval_time, ea_time=ea_time)
