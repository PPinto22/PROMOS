import MultiNEAT as NEAT


# def get_params():
#     params = NEAT.Parameters()
#     params.PopulationSize = 3
#     params.DynamicCompatibility = True
#     params.NormalizeGenomeSize = True
#     params.WeightDiffCoeff = 0.1
#     params.CompatTreshold = 2.0
#     params.YoungAgeTreshold = 15
#     params.SpeciesMaxStagnation = 15
#     params.OldAgeTreshold = 35
#     params.MinSpecies = 2
#     params.MaxSpecies = 10
#     params.RouletteWheelSelection = False
#     params.RecurrentProb = 0.0
#     params.OverallMutationRate = 1.0
#
#     params.ArchiveEnforcement = False
#
#     params.MutateWeightsProb = 0.05
#
#     params.WeightMutationMaxPower = 0.5
#     params.WeightReplacementMaxPower = 8.0
#     params.MutateWeightsSevereProb = 0.0
#     params.WeightMutationRate = 0.25
#     params.WeightReplacementRate = 0.9
#
#     params.MaxWeight = 8
#
#     params.MutateAddNeuronProb = 0.001
#     params.MutateAddLinkProb = 0.3
#     params.MutateRemLinkProb = 0.0
#
#     params.MinActivationA = 1
#     params.MaxActivationA = 1
#
#     params.ActivationFunction_SignedSigmoid_Prob = 0.0
#     params.ActivationFunction_UnsignedSigmoid_Prob = 1.0
#     params.ActivationFunction_Tanh_Prob = 0.0
#     params.ActivationFunction_SignedStep_Prob = 0.0
#
#     params.CrossoverRate = 0.4
#     params.MultipointCrossoverRate = 0.0
#     params.SurvivalRate = 0.2
#
#     params.MutateNeuronTraitsProb = 0
#     params.MutateLinkTraitsProb = 0
#
#     params.AllowLoops = True
#     params.AllowClones = True
#
#     return params

def get_params():
    params = NEAT.Parameters()
    params.PopulationSize = 25
    params.DynamicCompatibility = True
    params.AllowClones = False
    params.CompatTreshold = 5.0
    params.CompatTresholdModifier = 0.3
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 100
    params.OldAgeTreshold = 35
    params.MinSpecies = 2
    params.MaxSpecies = 5
    params.RouletteWheelSelection = True
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 0.02
    params.MutateWeightsProb = 0.90
    params.WeightMutationMaxPower = 1.0
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.5
    params.WeightMutationRate = 0.75
    params.MaxWeight = 20
    params.MutateAddNeuronProb = 0.01
    params.MutateAddLinkProb = 0.02
    params.MutateRemLinkProb = 0.00
    params.Elitism = 0.1
    params.CrossoverRate = 0.5
    params.MutateWeightsSevereProb = 0.01

    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0

    return params