from enum import Enum
import configparser


class BloatType(Enum):
    TIME = 'time'
    CONNECTIONS = 'connections'


class PenaltyFunction(Enum):
    STEP = 'step'
    POPMAX = 'popmax'


class FitnessOptions:
    def __init__(self, config):
        self.bloat_type = BloatType(config['bloat_type'])
        self.penalty_function = PenaltyFunction(config['penalty_function'])
        if self.penalty_function is PenaltyFunction.STEP:
            self.lower_limit = float(config['lower_limit'])
            self.step_width = float(config['step_width'])
            self.step_penalty_increase = float(config['step_penalty_increase'])
            assert self.step_width > 0 and self.step_penalty_increase > 0
        elif self.penalty_function is PenaltyFunction.POPMAX:
            self.popmax_alpha = float(config.get('popmax_alpha', 0.1))


class MutationOptions:
    def __init__(self, config):
        self.bloat_type = BloatType(config['bloat_type'])
        self.upper_limit = float(config['upper_limit'])
        self.lower_limit = float(config.get('lower_limit', self.upper_limit))
        self.mut_neurons_delta = float(config['mut_neurons_delta'])  # percentage
        self.mut_connections_delta = float(config['mut_connections_delta'])  # percentage
        self.frequency = int(config['frequency'])

        assert self.upper_limit >= self.lower_limit
        assert self.mut_neurons_delta >= 0 and self.mut_connections_delta >= 0 and self.frequency >= 1


class BloatOptions:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        if not config.read(config_file):
            raise ValueError('Invalid config file: {}'.format(config_file))

        self.mutation_options = MutationOptions(config['mutations']) if config.has_section('mutations') else None
        self.fitness_options = FitnessOptions(config['fitness']) if config.has_section('fitness') else None

    @classmethod
    def has_mutation_options(cls, bloat_options):
        return bloat_options is not None and bloat_options.mutation_options is not None

    @classmethod
    def has_fitness_options(cls, bloat_options):
        return bloat_options is not None and bloat_options.fitness_options is not None


class FitnessAdjuster:
    def __init__(self, fitness_options):
        assert isinstance(fitness_options, FitnessOptions)
        self.options = fitness_options

    def _step(self, evaluation):
        state = self.get_bloat_state(evaluation)

        if state < self.options.lower_limit:
            return evaluation.fitness  # Do not penalize

        distance = state - self.options.lower_limit
        steps = (distance // self.options.step_width) + 1
        penalty = 1 + (steps * self.options.step_penalty_increase)
        return evaluation.fitness / penalty

    def _popmax(self, evaluation, pop_max):
        state = self.get_bloat_state(evaluation)
        return evaluation.fitness * ((pop_max / state) ** self.options.popmax_alpha)

    def get_bloat_state(self, evaluation):
        if self.options.bloat_type is BloatType.CONNECTIONS:
            return evaluation.connections
        elif self.options.bloat_type is BloatType.TIME:
            return evaluation.eval_time  # microseconds
        else:
            raise AttributeError("Invalid bloat type: {}".format(self.options.bloat_type))

    def get_adjusted_fitness(self, evaluation, pop_max=None):
        f = self.options.penalty_function
        if f is PenaltyFunction.STEP:
            fitness_adj = self._step(evaluation)
        elif f is PenaltyFunction.POPMAX:
            assert pop_max is not None
            fitness_adj = self._popmax(evaluation, pop_max)
        else:
            raise NotImplementedError(
                "Penalty function {} is not implemented".format(self.options.penalty_function))
        return fitness_adj

    def get_pop_adjusted_fitness(self, pop_evaluations):
        f = self.options.penalty_function
        pop_max = max(self.get_bloat_state(e) for e in pop_evaluations) if f is PenaltyFunction.POPMAX else None

        return [self.get_adjusted_fitness(e, pop_max) for e in pop_evaluations]

    @classmethod
    def maybe_get_pop_adjusted_fitness(cls, adjuster, evaluation_list):
        if adjuster is None:
            return [e.fitness for e in evaluation_list]
        else:
            return adjuster.get_pop_adjusted_fitness(evaluation_list)


class MutationRateController:
    class Phase(Enum):
        DEFAULT = 0
        SIMPLIF = 1

    def __init__(self, params, mutation_options):
        self.params = params
        self.base_add_neuron = params.MutateAddNeuronProb
        self.base_rem_neuron = params.MutateRemSimpleNeuronProb
        self.base_mut_neuron = self.base_add_neuron + self.base_rem_neuron
        self.base_add_link = params.MutateAddLinkProb
        self.base_rem_link = params.MutateRemLinkProb
        self.base_mut_link = self.base_add_link + self.base_rem_link

        assert isinstance(mutation_options, MutationOptions)
        self.options = mutation_options

        self.mut_neurons_delta = self.options.mut_neurons_delta/100*self.base_mut_neuron
        self.mut_connections_delta = self.options.mut_connections_delta / 100 * self.base_mut_link

        self.phase = MutationRateController.Phase.DEFAULT

    def set_params(self, params, set_base_probs=True):
        self.params = params
        if set_base_probs:
            self.base_add_neuron = params.MutateAddNeuronProb
            self.base_rem_neuron = params.MutateRemSimpleNeuronProb
            self.base_mut_neuron = self.base_add_neuron + self.base_rem_neuron
            self.base_add_link = params.MutateAddLinkProb
            self.base_rem_link = params.MutateRemLinkProb
            self.base_mut_link = self.base_add_link + self.base_rem_link

    def reset(self):
        self.phase = MutationRateController.Phase.DEFAULT

    def simplify(self):
        neuron_delta = min(self.params.MutateAddNeuronProb, self.mut_neurons_delta)
        link_delta = min(self.params.MutateAddLinkProb, self.mut_connections_delta)

        self.params.MutateAddNeuronProb -= neuron_delta
        self.params.MutateRemSimpleNeuronProb += neuron_delta
        self.params.MutateAddLinkProb -= link_delta
        self.params.MutateRemLinkProb += link_delta

    def begin_simplification(self):
        self.phase = MutationRateController.Phase.SIMPLIF
        self.simplify()

    def begin_default_phase(self):
        self.phase = MutationRateController.Phase.DEFAULT
        self.params.MutateAddNeuronProb = self.base_add_neuron
        self.params.MutateRemSimpleNeuronProb = self.base_rem_neuron
        self.params.MutateAddLinkProb = self.base_add_link
        self.params.MutateRemLinkProb = self.base_rem_link

    def adjust(self, state, generation=None):
        if generation is not None and generation % self.options.frequency != 0:
            return  # Only adjust every [self.options.frequency] generations

        if self.phase is MutationRateController.Phase.DEFAULT:
            if state < self.options.upper_limit:
                pass  # Normal state, do nothing
            else:
                self.begin_simplification()
        elif self.phase is MutationRateController.Phase.SIMPLIF:
            if state > self.options.upper_limit:
                self.simplify()
            elif state > self.options.lower_limit:
                pass
            else:
                self.begin_default_phase()
        else:
            raise AttributeError("Invalid phase")
