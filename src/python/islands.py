import datetime
import multiprocessing as mp
import os
from functools import partial
from multiprocessing import Queue
from time import sleep

import numpy as np
from typing import Dict
import math

import MultiNEAT as neat

import evaluators
from params import get_params, ParametersWrapper
import util

N_ISLANDS = 5
MIGRATION_FRACTION = 0.2  # Percentage of the population to migrate
MIGRATION_FREQUENCY = 1  # Generations between migrations
GENERATIONS = 50
PARAMS = get_params()

LIST_EVALUATOR = evaluators.evaluate_genome_list_parallel
EVALUATION_PROCESSES = os.cpu_count() or 1
GENOME_EVALUATOR = evaluators.evaluate_auc

DATA_FILE_PATH = '../../data/data.csv'
OUT_DIR = '../../results'


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
        best_genomes = self.population.GetBestGenomesBySpecies(math.ceil(MIGRATION_FRACTION * PARAMS.PopulationSize))
        print("[DEBUG:Island_{}] Sending {} genomes".format(self.id_, len(best_genomes)))
        self.dest_q.put(best_genomes)

    def receive_migration(self):
        print("[DEBUG:Island_{}] Waiting for genomes".format(self.id_))
        genomes = self.recep_q.get()
        print("[DEBUG:Island_{}] Received {} genomes".format(self.id_, len(genomes)))
        self.population.ReplaceGenomes(genomes)
        print("[DEBUG:Island_{}] Genome replacement successful".format(self.id_))

    def run(self):
        g = neat.Genome(0, 11, 0, 1, False, neat.ActivationFunction.UNSIGNED_SIGMOID,
                        neat.ActivationFunction.UNSIGNED_SIGMOID, 0, PARAMS, 5)
        self.population = neat.Population(g, PARAMS, True, 1.0, self.id_)

        all_time_best = None
        for generation in range(GENERATIONS):
            print("[DEBUG:Island_{}] Starting generation {}".format(self.id_, generation))

            genome_list = util.get_genome_list(self.population)
            evaluation_list = LIST_EVALUATOR(genome_list, partial(GENOME_EVALUATOR, data=data, true_targets=true_targets))

            best_evaluation = max(evaluation_list, key=lambda e: e.fitness)
            print("[DEBUG:Island_{}] Best fitness of generation {}: {}".format(self.id_,
                                                                               generation,
                                                                               best_evaluation.fitness))
            if all_time_best is None or best_evaluation.fitness > all_time_best.fitness:
                all_time_best = best_evaluation
                all_time_best.save_genome_copy()
                print("[DEBUG:Island_{}] Sending new best to master".format(self.id_))
                self.master_q.put(all_time_best)
                print("[DEBUG:Island_{}] New best sent to master".format(self.id_))

            if generation > 0 and generation % MIGRATION_FREQUENCY == 0:
                self.send_migration()
                self.receive_migration()

            print("[DEBUG:Island_{}] Calling Epoch()".format(self.id_))
            self.population.Epoch()
            print("[DEBUG:Island_{}] Epoch() finished".format(self.id_))

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

    def setup(self):
        # Initialize islands and setup data for evaluation
        for i in range(N_ISLANDS):
            island = Island(i, self.queue)
            island.set_data(self.data, self.true_targets)
            self.islands[i] = island
        # Setup ring topology
        for i in range(N_ISLANDS):
            destination = self.islands[(i + 1) % len(self.islands)]
            self.islands[i].set_destination(destination)

    def run(self):
        initial_time = datetime.datetime.now()
        finished = 0

        for island in self.islands.values():
            island.start()
        while finished < N_ISLANDS:
            message = self.queue.get()
            if isinstance(message, evaluators.GenomeEvaluation):
                if self.best is None or message.fitness > self.best.fitness:
                    self.best = message
            elif isinstance(message, Message):
                if message.msg == 'Finished':
                    print("[DEBUG:Master] Received termination from island {}".format(message.sender))
                    finished += 1

        elapsed_time = datetime.datetime.now() - initial_time
        self.best = GENOME_EVALUATOR(self.best.genome, self.data, self.true_targets)
        util.write_results(
            out_file_path='{}/neat_islands_{}.json'.format(OUT_DIR, util.get_current_datetime_string()),
            best_evaluation=self.best,

            params=ParametersWrapper(PARAMS),
            islands=N_ISLANDS,
            generations=GENERATIONS,
            run_time=datetime.datetime.now() - initial_time
        )


if __name__ == '__main__':
    data = util.read_data(DATA_FILE_PATH)
    true_targets = np.array([row['target'] for row in data])

    master = Master(data, true_targets)
    master.setup()
    master.run()
