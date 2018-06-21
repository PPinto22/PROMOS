import MultiNEAT as neat
from enum import Enum


class BloatController:
    class Phase(Enum):
        DEFAULT = 0
        SIMPLIF = 1

    def __init__(self, params, upper_limit, lower_limit, neuron_mut_delta, link_mut_delta):
        assert upper_limit >= lower_limit
        assert neuron_mut_delta >= 0 and link_mut_delta >= 0

        self.base_add_neuron = params.MutateAddNeuronProb
        self.base_rem_neuron = params.MutateRemSimpleNeuronProb
        self.base_mut_neuron = self.base_add_neuron + self.base_rem_neuron
        self.base_add_link = params.MutateAddLinkProb
        self.base_rem_link = params.MutateRemLinkProb
        self.base_mut_link = self.base_add_link + self.base_rem_link

        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.neuron_mut_delta = neuron_mut_delta
        self.link_mut_delta = link_mut_delta

        self.phase = BloatController.Phase.DEFAULT

    def simplify(self, params):
        # neuron_delta1 = min(self.base_mut_neuron - params.MutateRemSimpleNeuronProb, self.neuron_mut_delta)
        # neuron_delta2 = min(self.base_mut_neuron - params.MutateAddNeuronProb, self.neuron_mut_delta)
        # neuron_delta = max(min(neuron_delta1, neuron_delta2), 0)
        #
        # link_delta1 = min(self.base_mut_link - params.MutateRemLinkProb, self.link_mut_delta)
        # link_delta2 = min(self.base_mut_link - params.MutateAddLinkProb, self.link_mut_delta)
        # link_delta = max(min(link_delta1, link_delta2), 0)

        neuron_delta = min(params.MutateAddNeuronProb, self.neuron_mut_delta)
        link_delta = min(params.MutateAddLinkProb, self.link_mut_delta)

        params.MutateAddNeuronProb -= neuron_delta
        params.MutateRemSimpleNeuronProb += neuron_delta
        params.MutateAddLinkProb -= link_delta
        params.MutateRemLinkProb += link_delta

    def begin_simplification(self, params):
        self.phase = BloatController.Phase.SIMPLIF
        self.simplify(params)

    def begin_default_phase(self, params):
        self.phase = BloatController.Phase.DEFAULT
        params.MutateAddNeuronProb = self.base_add_neuron
        params.MutateRemSimpleNeuronProb = self.base_rem_neuron
        params.MutateAddLinkProb = self.base_add_link
        params.MutateRemLinkProb = self.base_rem_link

    def adjust(self, params, state):
        if self.phase is BloatController.Phase.DEFAULT:
            if state < self.upper_limit:
                pass  # Normal state, do nothing
            else:
                self.begin_simplification(params)  # Begin simplification
        elif self.phase is BloatController.Phase.SIMPLIF:
            if state > self.upper_limit:
                self.simplify(params)
            elif state > self.lower_limit:
                pass
            else:
                self.begin_default_phase(params)
        else:
            raise AttributeError("Invalid phase")




if __name__ == '__main__':
    class P:
        def __init__(self):
            self.MutateAddLinkProb = 0.5
            self.MutateRemLinkProb = 0.1
            self.MutateAddNeuronProb = 0.08
            self.MutateRemSimpleNeuronProb = 0.02

    params = P()

    controller = BloatController(params, 100, 80, 0.2, 0.1)
    controller.adjust(params, 0)
    controller.adjust(params, 80)
    controller.adjust(params, 101)
    controller.adjust(params, 100.1)
    controller.adjust(params, 100.1)
    controller.adjust(params, 100.1)
    controller.adjust(params, 100.1)
    controller.adjust(params, 95)
    controller.adjust(params, 81)
    controller.adjust(params, 80)
    controller.adjust(params, 90)
    controller.adjust(params, 120)