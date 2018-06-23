from enum import Enum
import configparser


class FitnessOptions:
    def __init__(self, config):
        pass  # TODO


class LimitBy(Enum):
    TIME = 'time'
    CONNECTIONS = 'connections'


class MutationOptions:
    def __init__(self, config):
        self.limit_by = LimitBy(config['limit_by'])
        self.upper_limit = float(config['upper_limit'])
        self.lower_limit = float(config.get('lower_limit', self.upper_limit))
        self.mut_neurons_delta = float(config['mut_neurons_delta'])
        self.mut_connections_delta = float(config['mut_connections_delta'])
        self.frequency = int(config['frequency'])

        assert self.upper_limit >= self.lower_limit
        assert self.mut_neurons_delta >= 0 and self.mut_connections_delta >= 0 and self.frequency >= 1


class BloatOptions:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        self.mutation_options = MutationOptions(config['mutations']) if config.has_section('mutations') else None
        self.fitness_options = FitnessOptions(config['fitness']) if config.has_section('fitness') else None

    @classmethod
    def has_mutation_options(cls, bloat_options):
        return bloat_options is not None and bloat_options.mutation_options is not None

    @classmethod
    def has_fitness_options(cls, bloat_options):
        return bloat_options is not None and bloat_options.fitness_options is not None


class BloatController:
    class Phase(Enum):
        DEFAULT = 0
        SIMPLIF = 1

    def __init__(self, params, mutation_options):
        self.base_add_neuron = params.MutateAddNeuronProb
        self.base_rem_neuron = params.MutateRemSimpleNeuronProb
        self.base_mut_neuron = self.base_add_neuron + self.base_rem_neuron
        self.base_add_link = params.MutateAddLinkProb
        self.base_rem_link = params.MutateRemLinkProb
        self.base_mut_link = self.base_add_link + self.base_rem_link

        self.upper_limit = mutation_options.upper_limit
        self.lower_limit = mutation_options.lower_limit
        self.neuron_mut_delta = mutation_options.mut_neurons_delta
        self.link_mut_delta = mutation_options.mut_connections_delta
        self.frequency = mutation_options.frequency

        self.phase = BloatController.Phase.DEFAULT

    def reset(self):
        self.phase = BloatController.Phase.DEFAULT

    def simplify(self, params):
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

    def adjust(self, params, state, generation=None):
        if generation is not None and generation % self.frequency != 0:
            return  # Only adjust every [self.frequency] generations

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


# if __name__ == '__main__':
#     options = BloatOptions('bloat.ini')

# if __name__ == '__main__':
#     class P:
#         def __init__(self):
#             self.MutateAddLinkProb = 0.5
#             self.MutateRemLinkProb = 0.1
#             self.MutateAddNeuronProb = 0.08
#             self.MutateRemSimpleNeuronProb = 0.02
#
#     params = P()
#
#     controller = BloatController(params, 100, 80, 0.2, 0.1)
#     controller.adjust(params, 0)
#     controller.adjust(params, 80)
#     controller.adjust(params, 101)
#     controller.adjust(params, 100.1)
#     controller.adjust(params, 100.1)
#     controller.adjust(params, 100.1)
#     controller.adjust(params, 100.1)
#     controller.adjust(params, 95)
#     controller.adjust(params, 81)
#     controller.adjust(params, 80)
#     controller.adjust(params, 90)
#     controller.adjust(params, 120)








class A:
    def __init__(self, a):
        self.a = a

class Ctrl:
    def __init__(self, a):
        self.a = a

class Evolver:
    def __init__(self):
        self.a = A(1)
        self.ctrl = Ctrl(self.a)

        print(self.a is self.ctrl.a)

        self.a = A(2)

        print(self.a.a)
        print(self.ctrl.a.a)
        print(self.a is self.ctrl.a)

if __name__ == '__main__':
    Evolver()





























