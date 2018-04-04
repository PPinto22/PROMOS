import csv
import time
from concurrent.futures import ProcessPoolExecutor


# Get all genomes from the population
def get_genome_list(pop):
    genome_list = []
    for s in pop.Species:
        for i in s.Individuals:
            genome_list.append(i)
    return genome_list


try:
    import networkx as nx


    def genome_to_nx(g):

        nts = g.GetNeuronTraits()
        lts = g.GetLinkTraits()
        gr = nx.DiGraph()

        for i, tp, traits in nts:
            gr.add_node(i, **traits)

        for inp, outp, traits in lts:
            gr.add_edge(inp, outp, **traits)

        gr.genome_traits = g.GetGenomeTraits()

        return gr
except:
    pass


def read_data(data_file_path):
    with open(data_file_path) as data_file:
        reader = csv.DictReader(data_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
        return list(reader)
