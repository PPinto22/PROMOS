#!/usr/bin/python3

import csv
import MultiNEAT as NEAT
from sklearn.metrics import roc_auc_score

from params import params
from util import GetGenomeList, EvaluateGenomeList_Serial

DATA_FILE_PATH = 'data/data.csv'


def read_data(data_file_path):
    with open(data_file_path) as data_file:
        reader = csv.DictReader(data_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        return list(reader)


def evaluate(genome, data, true_targets):
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    predictions = []
    print("Starting evaluation")
    for row in data:
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
        predictions.append(output)
    print("Finished evaluation")
    print("Calculating auc score")
    auc = roc_auc_score(true_targets, predictions)
    print("AUC: {}".format(auc))
    return auc


if __name__ == '__main__':
    data = read_data(DATA_FILE_PATH)
    true_targets = [row['target'] for row in data]

    g = NEAT.Genome(0, 11, 0, 1, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID,
                    NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 5)
    pop = NEAT.Population(g, params, True, 1.0, 0)  # 0 is the RNG seed
    # pop.RNG.Seed(int(time.clock() * 100))

    for generation in range(10):
        genome_list = GetGenomeList(pop)
        fitness_list = EvaluateGenomeList_Serial(genome_list=genome_list,
                                                      evaluator=lambda genome: evaluate(genome, data, true_targets),
                                                      display=False)
        NEAT.ZipFitness(genome_list, fitness_list)
        pop.Epoch()
        best = max(fitness_list)
        print(best)
