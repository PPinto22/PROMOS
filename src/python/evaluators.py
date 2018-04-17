import MultiNEAT

import numpy as np
from collections import namedtuple

from sklearn.metrics import roc_curve, auc


class GenomeEvaluation:
    def __init__(self, genome, fitness, network, metrics):
        self.genome = genome
        self.network = network
        self.fitness = fitness
        self.metrics = metrics


class Metrics:
    ROC = namedtuple('ROC', 'fpr tpr thresholds auc')

    def __init__(self, roc_fpr, roc_tpr, roc_thresholds, roc_auc):
        self.roc = self.ROC(roc_fpr, roc_tpr, roc_thresholds, roc_auc)


def evaluate_auc(genome, data, true_targets):
    # print("[DEBUG] Genome ID: {}".format(genome.GetID()))
    net = MultiNEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    predictions = np.zeros(len(data))
    # print("[DEBUG] Starting evaluation")
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
                # 1  # Bias
            ]
        )
        net.Activate()
        output = net.Output()
        predictions[i] = output[0]
    net.Flush()

    fpr, tpr, thresholds = roc_curve(true_targets, predictions)
    roc_auc = auc(fpr, tpr)
    genome.SetFitness(roc_auc)
    genome.SetEvaluated()
    # print("[DEBUG] Fitness: {}".format(roc_auc))

    return GenomeEvaluation(genome, roc_auc, net, Metrics(fpr, tpr, thresholds, roc_auc))


def evaluate_genome_list_serial(genome_list, evaluator):
    return [evaluator(genome) for genome in genome_list]


# TODO: Não testado
# def evaluate_genome_list_parallel(genome_list, evaluator,
#                                   cores=8, display=True, ipython_client=None):
#     ''' If ipython_client is None, will use concurrent.futures.
#     Pass an instance of Client() in order to use an IPython cluster '''
#     evaluation_list = []
#     curtime = time.time()
#
#     if ipython_client is None or not ipython_installed:
#         with ProcessPoolExecutor(max_workers=cores) as executor:
#             for i, (f, net, metrics) in enumerate(executor.map(evaluator, genome_list)):
#                 evaluation_list += [GenomeEvaluation(genome_list[i], f, net, metrics)]
#
#                 if display:
#                     if ipython_installed: clear_output(wait=True)
#                     print('Individuals: (%s/%s) Fitness: %3.4f' % (i, len(genome_list), f))
#     else:
#         if type(ipython_client) == Client:
#             lbview = ipython_client.load_balanced_view()
#             amr = lbview.map(evaluator, genome_list, ordered=True, block=False)
#             for i, (f, net, metrics) in enumerate(amr):
#                 if display:
#                     if ipython_installed:
#                         clear_output(wait=True)
#                     print('Individual:', i, 'Fitness:', f)
#                 evaluation_list.append(GenomeEvaluation(genome_list[i], f, net, metrics))
#         else:
#             raise ValueError('Please provide valid IPython.parallel Client() as ipython_client')
#
#     elapsed = time.time() - curtime
#
#     if display:
#         print('seconds elapsed: %3.4f' % elapsed)
#
#     return evaluation_list
