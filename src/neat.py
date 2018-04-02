#!/usr/bin/python3

import csv
import MultiNEAT as NEAT

import cv2
import numpy

from params import params
from util import GetGenomeList, EvaluateGenomeList_Serial, ZipFitness
from viz import Draw

DATA_FILE_PATH = '../data/data.csv'


def read_data(data_file_path):
    with open(data_file_path) as data_file:
        reader = csv.DictReader(data_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        return list(reader)


def evaluate(genome, data, true_targets):
    print("[DEBUG] Genome ID: {}".format(genome.GetID()))
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    predictions = numpy.zeros(len(data))
    print("Starting evaluation")
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
    print("Finished evaluation")
    # print("Calculating auc score")
    # auc = roc_auc_score(true_targets, predictions)
    # print("AUC: {}".format(auc))
    # return auc
    return 1 / sum(numpy.subtract(true_targets, predictions) ** 2)


if __name__ == '__main__':
    data = read_data(DATA_FILE_PATH)
    true_targets = numpy.array([row['target'] for row in data])

    g = NEAT.Genome(0, 11, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 5)
    pop = NEAT.Population(g, params, True, 1.0, 0)  # 0 is the RNG seed
    # pop.RNG.Seed(int(time.clock() * 100))

    for generation in range(1000):
        print("--- Generation {} ---".format(generation))
        genome_list = GetGenomeList(pop)
        fitness_list = EvaluateGenomeList_Serial(genome_list=genome_list,
                                                 evaluator=lambda genome: evaluate(genome, data, true_targets),
                                                 display=False)
        ZipFitness(genome_list, fitness_list)

        best = max(fitness_list)
        best_genome = genome_list[fitness_list.index(best)]
        print("[DEBUG] Best fitness of generation {}: {}".format(generation, best))

        net = NEAT.NeuralNetwork()
        best_genome.BuildPhenotype(net)
        cv2.imshow("Best Genome", Draw(net))
        cv2.waitKey(1)

        pop.Epoch()
