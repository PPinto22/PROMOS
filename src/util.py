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

try:
    from IPython.display import clear_output
    from ipyparallel import Client

    ipython_installed = True
except:
    ipython_installed = False

try:
    from progressbar import ProgressBar

    pbar_installed = True
except:
    pbar_installed = False


class GenomeEvaluation:
    def __init__(self, genome, fitness, metrics):
        self.genome = genome
        self.fitness = fitness
        self.metrics = metrics


# Evaluates all genomes in sequential manner (using only 1 process) and
# returns a list of 'GenomeEvaluation' objects.
# evaluator is a callable that is supposed to take Genome as argument and
# return a pair (double fitness, object metrics), where metrics is an optional (can be None) object,
# defined by the user, that wraps any other relevant characteristics of the genome besides its fitness
def evaluate_genome_list_serial(genome_list, evaluator, display=True, show_elapsed=False):
    evaluation_list = []
    count = 0

    if display and show_elapsed:
        curtime = time.time()

    if display and pbar_installed:
        pbar = ProgressBar()
        pbar.max_value = len(genome_list)
        pbar.min_value = 0

    for i, g in enumerate(genome_list):
        f, metrics = evaluator(g)
        g.SetFitness(f)
        g.SetEvaluated()

        evaluation = GenomeEvaluation(g, f, metrics)
        evaluation_list.append(evaluation)

        if display:
            if not pbar_installed:
                if ipython_installed:
                    clear_output(wait=True)
                print('Individuals: (%s/%s) Fitness: %3.4f' % (count, len(genome_list), f))
            else:
                pbar.update(i)
        count += 1

    if display and pbar_installed:
        pbar.finish()

    if display and show_elapsed:
        elapsed = time.time() - curtime
        print('seconds elapsed: %s' % elapsed)

    return evaluation_list


# Evaluates all genomes in parallel manner (many processes) and returns a
# returns a list of 'GenomeEvaluation' objects.
# evaluator is a callable that is supposed to take Genome as argument and
# return a pair (double fitness, object metrics), where metrics is an optional (can be None) object,
# defined by the user, that wraps any other relevant characteristics of the genome besides its fitness
# TODO: Alterado e não testado
def evaluate_genome_list_parallel(genome_list, evaluator,
                                  cores=8, display=True, ipython_client=None):
    ''' If ipython_client is None, will use concurrent.futures.
    Pass an instance of Client() in order to use an IPython cluster '''
    evaluation_list = []
    curtime = time.time()

    if ipython_client is None or not ipython_installed:
        with ProcessPoolExecutor(max_workers=cores) as executor:
            for i, evaluation in enumerate(executor.map(evaluator, genome_list)):
                evaluation_list += [evaluation]

                if display:
                    if ipython_installed: clear_output(wait=True)
                    print('Individuals: (%s/%s) Fitness: %3.4f' % (i, len(genome_list), evaluation.fitness))
    else:
        if type(ipython_client) == Client:
            lbview = ipython_client.load_balanced_view()
            amr = lbview.map(evaluator, genome_list, ordered=True, block=False)
            for i, evaluation in enumerate(amr):
                if display:
                    if ipython_installed:
                        clear_output(wait=True)
                    print('Individual:', i, 'Fitness:', evaluation.fitness)
                evaluation_list.append(evaluation)
        else:
            raise ValueError('Please provide valid IPython.parallel Client() as ipython_client')

    elapsed = time.time() - curtime

    if display:
        print('seconds elapsed: %3.4f' % elapsed)

    return evaluation_list
