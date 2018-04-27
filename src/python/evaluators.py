import MultiNEAT as neat
import multiprocessing

import numpy as np
from collections import namedtuple

from sklearn.metrics import roc_curve, auc

import util


class GenomeEvaluation:
    def __init__(self, genome, fitness, metrics):
        self.genome = genome
        self.fitness = fitness
        self.metrics = metrics

    def save_genome_copy(self):
        self.genome = neat.Genome(self.genome)


ROC = namedtuple('ROC', 'fpr tpr thresholds auc')


class Metrics:
    def __init__(self, roc_fpr, roc_tpr, roc_thresholds, roc_auc):
        self.roc = ROC(roc_fpr, roc_tpr, roc_thresholds, roc_auc)


def predict(net, data):
    predictions = np.zeros(len(data))
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
    net.Flush()
    return predictions


def evaluate_auc(genome, data, true_targets, **kwargs):
    net = util.build_network(genome)

    predictions = predict(net, data)
    fpr, tpr, thresholds = roc_curve(true_targets, predictions)
    roc_auc = auc(fpr, tpr)
    genome.SetFitness(roc_auc)
    genome.SetEvaluated()

    return GenomeEvaluation(genome, roc_auc, None)


def evaluate_error(genome, data, true_targets, **kwargs):
    net = util.build_network(genome)

    predictions = predict(net, data)
    fitness = 1 / sum((abs(pred - target) for pred, target in zip(predictions, true_targets)))

    genome.SetFitness(fitness)
    genome.SetEvaluated()

    return GenomeEvaluation(genome, fitness, None)


def evaluate_genome_list_serial(genome_list, evaluator):
    return [evaluator(genome) for genome in genome_list]


def evaluate_genome_list_parallel(genome_list, evaluator, processes=None):
    with multiprocessing.Pool(processes=processes) as pool:
        evaluation_list = pool.map(evaluator, genome_list)

    for genome, eval in zip(genome_list, evaluation_list):
        genome.SetFitness(eval.fitness)
        genome.SetEvaluated()

    return evaluation_list
