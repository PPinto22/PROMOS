import MultiNEAT as neat
import datetime
import multiprocessing

import numpy as np

from sklearn.metrics import roc_curve, auc

import util


class GenomeEvaluation:
    def __init__(self, genome, fitness, neurons=None, connections=None, global_time=None, **kwargs):
        """
        :type connections: int
        :type neurons: int
        ;type global_time: datetime.timedelta
        """
        self.genome = genome
        self.genome_id = genome.GetID()
        self.fitness = fitness
        self.neurons = neurons
        self.connections = connections
        self.global_time = global_time
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def save_genome_copy(self):
        self.genome = neat.Genome(self.genome)


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
        net.FeedForward()
        output = net.Output()
        predictions[i] = output[0]
    net.Flush()
    return predictions


def evaluate_auc(genome, data, true_targets, initial_time=None, **kwargs):
    net = util.build_network(genome, **kwargs)

    predictions = predict(net, data)
    fpr, tpr, thresholds = roc_curve(true_targets, predictions)
    roc_auc = auc(fpr, tpr)
    genome.SetFitness(roc_auc)
    genome.SetEvaluated()

    time = datetime.datetime.now() - initial_time if initial_time is not None else None

    return GenomeEvaluation(genome, roc_auc,
                            neurons=len(util.get_network_neurons(net)),
                            connections=len(util.get_network_connections(net)),
                            global_time=time)


def evaluate_error(genome, data, true_targets, **kwargs):
    net = util.build_network(genome, **kwargs)

    predictions = predict(net, data)
    fitness = 1 / sum((abs(pred - target) for pred, target in zip(predictions, true_targets)))

    genome.SetFitness(fitness)
    genome.SetEvaluated()

    return GenomeEvaluation(genome, fitness,
                            neurons=len(util.get_network_neurons(net)),
                            connections=len(util.get_network_connections(net)))


def evaluate_genome_list_serial(genome_list, evaluator, **kwargs):
    return [evaluator(genome) for genome in genome_list]


def evaluate_genome_list_parallel(genome_list, evaluator, processes=None):
    with multiprocessing.Pool(processes=processes) as pool:
        evaluation_list = pool.map(evaluator, genome_list)

    for genome, eval in zip(genome_list, evaluation_list):
        genome.SetFitness(eval.fitness)
        genome.SetEvaluated()

    return evaluation_list
