#!/usr/bin/python3

import MultiNEAT as NEAT

import csv
from collections import namedtuple

import cv2
import numpy
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

from params import params
from util import get_genome_list, evaluate_genome_list_serial
from viz import Draw

DATA_FILE_PATH = '../data/data.csv'


def read_data(data_file_path):
    with open(data_file_path) as data_file:
        reader = csv.DictReader(data_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        return list(reader)


class Metrics:
    ROC = namedtuple('ROC', 'fpr tpr thresholds auc')

    def __init__(self, roc_fpr, roc_tpr, roc_thresholds, roc_auc):
        self.roc = self.ROC(roc_fpr, roc_tpr, roc_thresholds, roc_auc)


def evaluate(genome, data, true_targets):
    print("[DEBUG] Genome ID: {}".format(genome.GetID()))
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    predictions = numpy.zeros(len(data))
    print("[DEBUG] Starting evaluation")
    for i, row in enumerate(data):
        net.Flush()
        net.Input(
            [
                row['regioncontinent'],
                row['idcampaign'],
                row['idpartner'],
                row['idverticaltype'],
                row['idbrowser'],
                row['idaffmanager'],
                row['idapplication'],
                row['idoperator'],
                row['accmanager'],
                row['country_name'],
                1  # Bias
            ]
        )
        net.Activate()
        output = net.Output()
        predictions[i] = output[0]
    print("[DEBUG] Processing ROC")
    fpr, tpr, thresholds = roc_curve(true_targets, predictions)
    roc_auc = auc(fpr, tpr)
    print("[DEBUG] Fitness: {}".format(roc_auc))
    return roc_auc, Metrics(fpr, tpr, thresholds, roc_auc)


if __name__ == '__main__':
    data = read_data(DATA_FILE_PATH)
    true_targets = numpy.array([row['target'] for row in data])

    g = NEAT.Genome(0, 11, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 5)
    pop = NEAT.Population(g, params, True, 1.0, 0)  # 0 is the RNG seed
    # pop.RNG.Seed(int(time.clock() * 100))

    for generation in range(1000):
        print("--- Generation {} ---".format(generation))
        genome_list = get_genome_list(pop)
        evaluation_list = evaluate_genome_list_serial(genome_list=genome_list,
                                                      evaluator=lambda genome: evaluate(genome, data, true_targets),
                                                      display=False)

        best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
        print("[DEBUG] Best fitness of generation {}: {}".format(generation, best_evaluation.fitness))

        # Plot network
        net = NEAT.NeuralNetwork()
        best_evaluation.genome.BuildPhenotype(net)
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
