import multiprocessing as mp
import numpy as np
from typing import Dict

import MultiNEAT as neat
from evaluators import evaluate_genome_list_serial, evaluate_auc
from params import params
from util import read_data, get_genome_list

N_ISLANDS = 4
DATA_FILE_PATH = '../data/data.csv'


# TODO: Migracoes entre ilhas e comunicacao entre ilhas e master
# TODO: Parametros especificos do modelo de ilhas (e.g. frequencia de migracoes)

class Island(mp.Process):
    def __init__(self, id_):
        super().__init__()
        self.id_ = id_
        self.recep_q = mp.Queue()  # For reception of genomes sent from other islands
        self.master_pipe = mp.Pipe()  # For communication with the master
        self.dest_q = None  # Send genomes to this destination
        self.data = None
        self.true_targets = None

    def set_destination(self, other_island):
        self.dest_q = other_island.recep_q

    def set_data(self, data, true_targets):
        self.data = data
        self.true_targets = true_targets

    def run(self):
        # FIXME: Parametros hardcoded
        g = neat.Genome(0, 11, 0, 1, False, neat.ActivationFunction.UNSIGNED_SIGMOID,
                        neat.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 5)
        pop = neat.Population(g, params, True, 1.0, self.id_)

        for generation in range(1000):
            genome_list = get_genome_list(pop)
            evaluation_list = evaluate_genome_list_serial(genome_list,
                                                          lambda genome: evaluate_auc(genome, data, true_targets))

            best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
            print("[DEBUG:Island_{}] Best fitness of generation {}: {}".format(self.id_,
                                                                               generation,
                                                                               best_evaluation.fitness))

            pop.Epoch()


class Master:
    islands = ...  # type: Dict[int, Island]
    best = ...  # type: evaluators.GenomeEvaluation

    def __init__(self):
        self.islands = {}
        self.best = None

    def setup(self, n_islands, data, true_targets):
        # Initialize islands and setup data for evaluation
        for i in range(n_islands):
            island = Island(i)
            island.set_data(data, true_targets)
            self.islands[i] = Island(i)
        # Setup ring topology
        for i in range(n_islands):
            destination = self.islands[(i + 1) % len(self.islands)]
            self.islands[i].set_destination(destination)

    def run(self):
        for island in self.islands.values():
            island.start()
        while True:
            pass  # TODO: Recolher informações das ilhas


if __name__ == '__main__':
    data = read_data(DATA_FILE_PATH)
    true_targets = np.array([row['target'] for row in data])

    master = Master()
    master.setup(N_ISLANDS, data, true_targets)
    master.run()
