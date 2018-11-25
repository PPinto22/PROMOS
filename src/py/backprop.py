import MultiNEAT as neat
from functools import partial

import numpy as np
import params
import util
from evaluator import Evaluator
import multiprocessing as mp
import concurrent.futures


def _sgd_partial(genome, *args, **kwargs):
    bp = Backprop(genome, *args, **kwargs)
    bp.sgd()
    return bp.get_weights_and_biases(bp.net)


class Backprop:
    @staticmethod
    def sgd_pop(genome_list, parallel=1, *args, **kwargs):
        if parallel == 1:
            for genome in genome_list:
                bp = Backprop(genome, *args, **kwargs)
                bp.sgd()
                bp.update_genome(genome)
        else:
            sgd_partial = partial(_sgd_partial, *args, **kwargs)
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=parallel)
            futures = [pool.submit(sgd_partial, genome) for genome in genome_list]
            concurrent.futures.wait(futures)
            nets = [util.try_(future.result) for future in futures]
            for genome, net in zip(genome_list, nets):
                if net is not None:
                    weights, biases = net
                    Backprop.set_weight_and_biases(genome, weights, biases)

    def __init__(self, genome, inputs, targets, lr=1, mini_batches=100, epochs=1, maxweight=20, maxbias=5):
        self.genome = genome
        self.net = util.build_network(genome)
        self.sorted_neurons, self.neuron_links = self.prepare()
        self.lr = lr  # learning rate
        self.mini_batches = mini_batches
        self.epochs = epochs
        self.maxweight = maxweight
        self.maxbias = maxbias
        self.inputs, self.targets, self.n, self.batch_len = None, None, None, None
        self.set_data(inputs, targets)

    def set_data(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        assert len(inputs) == len(targets)
        assert len(inputs) > self.mini_batches
        self.n = len(self.inputs)
        self.batch_len = self.n // self.mini_batches

    def prepare(self):
        """
        :return:
            - sorted_neurons, a list of neuron indexes sorted backwards (from target to input);
            - neurons_links: Map<neuron_id, (list[(index, incoming_link)], list[(index, outgoing_link)])>
        """
        sorted_neurons = []
        neuron_links = {
            neuron_idx: (
                [(i, link) for i, link in enumerate(self.net.connections) if link.target_neuron_idx == neuron_idx],
                [(i, link) for i, link in enumerate(self.net.connections) if link.source_neuron_idx == neuron_idx])
            for neuron_idx, neuron in enumerate(self.net.neurons)}

        # Mutable auxiliary dict
        neuron_outgoing = {neuron_idx: [link for link in self.net.connections if link.source_neuron_idx == neuron_idx]
                           for neuron_idx, neuron in enumerate(self.net.neurons) if
                           neuron.type != neat.NeuronType.INPUT}
        while neuron_outgoing:
            for neuron, outgoing in neuron_outgoing.items():
                if not outgoing:
                    free_neuron = neuron
                    for source_neuron in [link.source_neuron_idx for (_, link) in neuron_links[neuron][0]
                                          if self.net.neurons[link.source_neuron_idx].type != neat.NeuronType.INPUT]:
                        neuron_outgoing[source_neuron] = [link for link in neuron_outgoing[source_neuron] if
                                                          link.target_neuron_idx != neuron]
                    break
            else:
                raise AttributeError("Something went wrong, possibly due to loops in the network.")
            del neuron_outgoing[free_neuron]
            sorted_neurons.append(free_neuron)

        return sorted_neurons, neuron_links

    def get_batches(self):
        batch_size = self.n // self.mini_batches
        for k in range(0, self.n, batch_size):
            yield zip(self.inputs[k:k + batch_size], self.targets[k:k + batch_size])

    def shuffle_data(self):
        pass  # TODO

    @staticmethod
    def update_genome(genome, net):
        for idx, link in enumerate(net.connections):
            genome.LinkGenes[idx].Weight = link.weight
        for idx, neuron in enumerate(net.neurons):
            genome.NeuronGenes[idx].Bias = neuron.bias

    @staticmethod
    def set_weight_and_biases(genome, weights, biases):
        for idx, weight in enumerate(weights):
            genome.LinkGenes[idx].Weight = weight
        for idx, bias in enumerate(biases):
            genome.NeuronGenes[idx].Bias = bias

    @staticmethod
    def get_weights_and_biases(net):
        weights = [link.weight for link in net.connections]
        biases = [neuron.bias for neuron in net.neurons]
        return weights, biases

    def sgd(self):
        for i in range(self.epochs):
            self.shuffle_data()
            for batch in self.get_batches():
                self.update_mini_batch(batch)

    def update_mini_batch(self, mini_batch):
        w_grads = {i: 0 for i in range(self.net.NumConnections())}
        b_grads = {i: 0 for i in range(self.net.NumNeurons())}
        for inputs, target in mini_batch:
            batch_w_grads, batch_b_grads = self.backprop(inputs, target)
            for idx, grad in batch_w_grads.items():
                w_grads[idx] += grad
            for idx, grad in batch_b_grads.items():
                b_grads[idx] += grad
        for idx, grad in w_grads.items():
            old_weight = self.net.connections[idx].weight
            self.net.connections[idx].weight = util.constraint(old_weight - grad * (self.lr / self.batch_len),
                                                               self.maxweight)
        for idx, grad in b_grads.items():
            old_bias = self.net.neurons[idx].bias
            self.net.neurons[idx].bias = util.constraint(old_bias - grad * (self.lr / self.batch_len), self.maxbias)

    def backprop(self, inputs, target):
        """
        :param inputs: a single case of inputs to propagate through the network
        :param target: the true target
        :return: A pair of dictionaries corresponding to the weights and biases gradients
        """
        w_grads = {}
        deltas = {}
        # Forward pass
        Evaluator.predict_single(self.net, inputs)
        # First step: output neuron
        output_index = self.sorted_neurons[0]
        delta = self.calculate_output_delta(target)
        deltas[output_index] = delta
        for idx, grad in self.calculate_gradients(output_index, delta).items():
            w_grads[idx] = grad
        # Propagate backwards
        for neuron in self.sorted_neurons[1:]:
            delta = self.calculate_delta(neuron, deltas)
            deltas[neuron] = delta
            for idx, grad in self.calculate_gradients(neuron, delta).items():
                w_grads[idx] = grad
        self.net.Flush()
        return w_grads, deltas

    def calculate_output_delta(self, target):
        output_idx = self.sorted_neurons[0]
        output_neuron = self.net.neurons[output_idx]
        cost_derivative = self.cost_derivative(output_neuron.activation, target)
        activ_derivative = util.afderiv(output_neuron.af, output_neuron.activesum)
        delta = cost_derivative * activ_derivative
        return delta

    def calculate_delta(self, neuron_idx, deltas):
        neuron = self.net.neurons[neuron_idx]
        activ_derivative = util.afderiv(neuron.af, neuron.activesum)
        delta = sum(link.weight * deltas[link.target_neuron_idx] for (_, link) in
                    self.neuron_links[neuron_idx][1]) * activ_derivative
        return delta

    def calculate_gradients(self, neuron_idx, delta):
        """
        For each connection incoming to the current neuron, multiply the activation of the source neuron by delta
        :return: Map<Connection_idx, gradient_value>
        """
        return {idx: self.net.neurons[link.source_neuron_idx].activation * delta for (idx, link) in
                self.neuron_links[neuron_idx][0]}

    def cost_derivative(self, output_activation, target):
        return 2 * (output_activation - target)

# if __name__ == '__main__':
#     p = params._default_params()
#     g = neat.Genome(0, 2, 0, 1,
#                     False, neat.ActivationFunction.UNSIGNED_SIGMOID, neat.ActivationFunction.UNSIGNED_SIGMOID, 0,
#                     params._default_params(), 0)
#     rng = neat.RNG()
#     innov = neat.InnovationDatabase()
#     innov.Init(g)
#     g.Mutate_AddNeuron(innov, p, rng)
#     g.Mutate_AddNeuron(innov, p, rng)
#     g.Mutate_AddNeuron(innov, p, rng)
#     g.Mutate_AddLink(innov, p, rng)
#     g.Mutate_AddLink(innov, p, rng)
#     g.Mutate_AddLink(innov, p, rng)
#     g.Mutate_AddLink(innov, p, rng)
#     g.Save("/home/pedro/Desktop/genome.txt")
#
#     genome = neat.Genome('/home/pedro/Desktop/genome.txt')
#
#     inputs = [[0.4, 0.3] for _ in range(2000)]
#     targets = [0.7] * 2000
#     backprop = Backprop(genome, inputs, targets, epochs=100)
#     print(Evaluator.predict_single(backprop.net, [0.4, 0.3]))
#     backprop.SGD()
#     print(Evaluator.predict_single(backprop.net, [0.4, 0.3]))
#
