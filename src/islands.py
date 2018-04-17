import datetime
import multiprocessing as mp
from multiprocessing import Queue

import numpy as np
from typing import Dict
import math

import MultiNEAT as neat
from evaluators import evaluate_genome_list_serial, evaluate_auc, GenomeEvaluation
from params import get_params
from util import read_data, get_genome_list, write_results, get_current_datetime_string

N_ISLANDS = 5
MIGRATION_SIZE = 0.2  # Percentage of the population to migrate (rounded up)
MIGRATION_FREQUENCY = 1  # Generations between migrations
GENERATIONS = 50
params = get_params()

DATA_FILE_PATH = '../data/data.csv'
OUT_DIR = '../results'


class Message:
    def __init__(self, sender, msg, generation):
        self.sender = sender
        self.msg = msg
        self.generation = generation


class Island(mp.Process):
    def __init__(self, id_, master_q):
        super().__init__()
        self.id_ = id_
        self.recep_q = mp.Queue()  # For reception of genomes sent from other islands
        self.master_q = master_q  # For communication with the master
        self.dest_q = None  # Send genomes to this destination
        self.data = None
        self.true_targets = None
        self.population = None

    def set_destination(self, other_island):
        self.dest_q = other_island.recep_q

    def set_data(self, data, true_targets):
        self.data = data
        self.true_targets = true_targets

    def send_migration(self):
        best_genomes = self.population.GetBestGenomesBySpecies(math.ceil(MIGRATION_SIZE * params.PopulationSize))
        print("[DEBUG:Island_{}] Sending {} genomes".format(self.id_, len(best_genomes)))
        self.dest_q.put(best_genomes)

    def receive_migration(self):
        print("[DEBUG:Island_{}] Waiting for genomes".format(self.id_))
        genomes = self.recep_q.get()
        print("[DEBUG:Island_{}] Received {} genomes".format(self.id_, len(genomes)))
        self.population.ReplaceGenomes(genomes)
        print("[DEBUG:Island_{}] Genome replacement successful".format(self.id_))

    def run(self):
        # FIXME: Parametros hardcoded
        g = neat.Genome(0, 11, 0, 1, False, neat.ActivationFunction.UNSIGNED_SIGMOID,
                        neat.ActivationFunction.UNSIGNED_SIGMOID, 0, params, 5)
        self.population = neat.Population(g, params, True, 1.0, self.id_)

        all_time_best = None
        for generation in range(GENERATIONS):
            if generation > 0 and generation % MIGRATION_FREQUENCY == 0:
                self.send_migration()
                self.receive_migration()

            genome_list = get_genome_list(self.population)
            evaluation_list = evaluate_genome_list_serial(genome_list,
                                                          lambda genome: evaluate_auc(genome, data, true_targets))

            best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
            print("[DEBUG:Island_{}] Best fitness of generation {}: {}".format(self.id_,
                                                                               generation,
                                                                               best_evaluation.fitness))
            if all_time_best is None or best_evaluation.fitness > all_time_best.fitness:
                all_time_best = best_evaluation
                all_time_best.network = None
                all_time_best.metrics = None
                self.master_q.put(all_time_best)

            self.population.Epoch()

        self.master_q.put(Message(self.id_, 'Finished', GENERATIONS))
        print("[DEBUG:Island_{}] Terminated".format(self.id_))


class Master:
    islands = ...  # type: Dict[int, Island]
    best = ...  # type: evaluators.GenomeEvaluation

    def __init__(self, data, true_targets):
        self.data = data
        self.true_targets = true_targets
        self.islands = {}
        self.best = None
        self.queue = mp.Queue()

    def setup(self, n_islands):
        # Initialize islands and setup data for evaluation
        for i in range(n_islands):
            island = Island(i, self.queue)
            island.set_data(self.data, self.true_targets)
            self.islands[i] = island
        # Setup ring topology
        for i in range(n_islands):
            destination = self.islands[(i + 1) % len(self.islands)]
            self.islands[i].set_destination(destination)

    def run(self):
        initial_time = datetime.datetime.now()
        finished = 0

        for island in self.islands.values():
            island.start()
        while finished < N_ISLANDS:
            message = self.queue.get()
            if isinstance(message, GenomeEvaluation):
                if self.best is None or message.fitness > self.best.fitness:
                    self.best = message
            elif isinstance(message, Message):
                if message.msg == 'Finished':
                    print("[DEBUG:Master] Received termination from island {}".format(message.sender))
                    finished += 1

        elapsed_time = datetime.datetime.now() - initial_time
        self.best = evaluate_auc(self.best.genome, self.data, self.true_targets)
        write_results('{}/neat_islands_{}.json'.format(OUT_DIR, get_current_datetime_string()),
                      'neat_islands_{}'.format(N_ISLANDS), GENERATIONS,
                      datetime.datetime.now() - initial_time, self.best, params)


if __name__ == '__main__':
    data = read_data(DATA_FILE_PATH)
    true_targets = np.array([row['target'] for row in data])

    master = Master(data, true_targets)
    master.setup(N_ISLANDS)
    master.run()
