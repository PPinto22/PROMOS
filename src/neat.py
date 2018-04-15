#!/usr/bin/python3

import MultiNEAT as neat

import cv2
import matplotlib.pyplot as plt
import numpy as np

from evaluators import evaluate_genome_list_serial, evaluate_auc
from params import params
from util import get_genome_list, read_data, get_network_connections
from viz import Draw

DATA_FILE_PATH = '../data/data.csv'


if __name__ == '__main__':
    data = read_data(DATA_FILE_PATH)
    true_targets = np.array([row['target'] for row in data])

    g = neat.Genome(0, 10, 0, 1, False, neat.ActivationFunction.UNSIGNED_SIGMOID,
                    neat.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 5)
    pop = neat.Population(g, params, True, 1.0, 0)  # 0 is the RNG seed
    # pop.RNG.Seed(int(time.clock() * 100))

    for generation in range(50):
        print("--- Generation {} ---".format(generation))
        genome_list = get_genome_list(pop)
        evaluation_list = evaluate_genome_list_serial(genome_list,
                                                      lambda genome: evaluate_auc(genome, data, true_targets))

        best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
        print("[DEBUG] Best fitness of generation {}: {}".format(generation, best_evaluation.fitness))

        # Plot network
        net = neat.NeuralNetwork()
        best_evaluation.genome.BuildPhenotype(net)
        print('\n'.join([str(x) for x in (get_network_connections(net))]))
        cv2.imshow("Best Network", Draw(net))
        cv2.waitKey(1)

        # Plot ROC
        roc = best_evaluation.metrics.roc
        fig = plt.gcf()
        fig.canvas.set_window_title('ROC Curve')
        plt.clf()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.005])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(roc.fpr, roc.tpr, color='darkorange',
                 lw=2, label='ROC Curve (area = %.2f)' % roc.auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('Generation {}\'s Best Network'.format(generation))
        plt.legend(loc="lower right")
        plt.draw()
        plt.pause(0.001)

        pop.Epoch()
