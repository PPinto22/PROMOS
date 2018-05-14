import MultiNEAT as neat


def get_params0():
    params = neat.Parameters()
    params.PopulationSize = 150
    params.DynamicCompatibility = True
    params.AllowClones = True
    params.CompatTreshold = 5.0
    params.CompatTresholdModifier = 0.3
    params.YoungAgeTreshold = 15
    params.SpeciesMaxStagnation = 100
    params.OldAgeTreshold = 35
    params.MinSpecies = 2
    params.MaxSpecies = 10
    params.RouletteWheelSelection = True
    params.RecurrentProb = 0.0
    params.OverallMutationRate = 0.1
    params.MutateWeightsProb = 0.90
    params.WeightMutationMaxPower = 1.0
    params.WeightReplacementMaxPower = 5.0
    params.MutateWeightsSevereProb = 0.01
    params.WeightMutationRate = 0.75
    params.MaxWeight = 20
    params.MutateAddNeuronProb = 0.05
    params.MutateAddLinkProb = 0.1
    params.MutateRemLinkProb = 0.05
    params.EliteFraction = 0.15
    params.CrossoverRate = 0.5

    params.ActivationFunction_UnsignedGauss_Prob = 0.25
    params.ActivationFunction_UnsignedSigmoid_Prob = 0.25
    params.ActivationFunction_UnsignedSine_Prob = 0.25
    params.ActivationFunction_Relu_Prob = 0.25

    params.MutateNeuronTraitsProb = 0
    params.MutateLinkTraitsProb = 0

    return params


params = [get_params0]


def get_params(i):
    return params[i]()


class ParametersWrapper:
    def __init__(self, neat_params):
        self.PopulationSize = neat_params.PopulationSize
        self.DynamicCompatibility = neat_params.DynamicCompatibility
        self.MinSpecies = neat_params.MinSpecies
        self.MaxSpecies = neat_params.MaxSpecies
        self.InnovationsForever = neat_params.InnovationsForever
        self.AllowClones = neat_params.AllowClones
        self.ArchiveEnforcement = neat_params.ArchiveEnforcement
        self.NormalizeGenomeSize = neat_params.NormalizeGenomeSize
        self.CustomConstraints = neat_params.CustomConstraints
        self.YoungAgeTreshold = neat_params.YoungAgeTreshold
        self.YoungAgeFitnessBoost = neat_params.YoungAgeFitnessBoost
        self.SpeciesDropoffAge = neat_params.SpeciesDropoffAge
        self.StagnationDelta = neat_params.StagnationDelta
        self.OldAgeTreshold = neat_params.OldAgeTreshold
        self.OldAgePenalty = neat_params.OldAgePenalty
        self.DetectCompetetiveCoevolutionStagnation = neat_params.DetectCompetetiveCoevolutionStagnation
        self.KillWorstSpeciesEach = neat_params.KillWorstSpeciesEach
        self.KillWorstAge = neat_params.KillWorstAge
        self.SurvivalRate = neat_params.SurvivalRate
        self.CrossoverRate = neat_params.CrossoverRate
        self.OverallMutationRate = neat_params.OverallMutationRate
        self.InterspeciesCrossoverRate = neat_params.InterspeciesCrossoverRate
        self.MultipointCrossoverRate = neat_params.MultipointCrossoverRate
        self.RouletteWheelSelection = neat_params.RouletteWheelSelection
        self.PhasedSearching = neat_params.PhasedSearching
        self.DeltaCoding = neat_params.DeltaCoding
        self.SimplifyingPhaseMPCTreshold = neat_params.SimplifyingPhaseMPCTreshold
        self.SimplifyingPhaseStagnationTreshold = neat_params.SimplifyingPhaseStagnationTreshold
        self.ComplexityFloorGenerations = neat_params.ComplexityFloorGenerations
        self.NoveltySearch_K = neat_params.NoveltySearch_K
        self.NoveltySearch_P_min = neat_params.NoveltySearch_P_min
        self.NoveltySearch_Dynamic_Pmin = neat_params.NoveltySearch_Dynamic_Pmin
        self.NoveltySearch_No_Archiving_Stagnation_Treshold = neat_params.NoveltySearch_No_Archiving_Stagnation_Treshold
        self.NoveltySearch_Pmin_lowering_multiplier = neat_params.NoveltySearch_Pmin_lowering_multiplier
        self.NoveltySearch_Pmin_min = neat_params.NoveltySearch_Pmin_min
        self.NoveltySearch_Quick_Archiving_Min_Evaluations = neat_params.NoveltySearch_Quick_Archiving_Min_Evaluations
        self.NoveltySearch_Pmin_raising_multiplier = neat_params.NoveltySearch_Pmin_raising_multiplier
        self.NoveltySearch_Recompute_Sparseness_Each = neat_params.NoveltySearch_Recompute_Sparseness_Each
        self.MutateAddNeuronProb = neat_params.MutateAddNeuronProb
        self.SplitRecurrent = neat_params.SplitRecurrent
        self.SplitLoopedRecurrent = neat_params.SplitLoopedRecurrent
        self.MutateAddLinkProb = neat_params.MutateAddLinkProb
        self.MutateAddLinkFromBiasProb = neat_params.MutateAddLinkFromBiasProb
        self.MutateRemLinkProb = neat_params.MutateRemLinkProb
        self.MutateRemSimpleNeuronProb = neat_params.MutateRemSimpleNeuronProb
        self.LinkTries = neat_params.LinkTries
        self.RecurrentProb = neat_params.RecurrentProb
        self.RecurrentLoopProb = neat_params.RecurrentLoopProb
        self.MutateWeightsProb = neat_params.MutateWeightsProb
        self.MutateWeightsSevereProb = neat_params.MutateWeightsSevereProb
        self.WeightMutationRate = neat_params.WeightMutationRate
        self.WeightMutationMaxPower = neat_params.WeightMutationMaxPower
        self.WeightReplacementRate = neat_params.WeightReplacementRate
        self.WeightReplacementMaxPower = neat_params.WeightReplacementMaxPower
        self.MaxWeight = neat_params.MaxWeight
        self.MutateActivationAProb = neat_params.MutateActivationAProb
        self.MutateActivationBProb = neat_params.MutateActivationBProb
        self.ActivationAMutationMaxPower = neat_params.ActivationAMutationMaxPower
        self.ActivationBMutationMaxPower = neat_params.ActivationBMutationMaxPower
        self.MinActivationA = neat_params.MinActivationA
        self.MaxActivationA = neat_params.MaxActivationA
        self.MinActivationB = neat_params.MinActivationB
        self.MaxActivationB = neat_params.MaxActivationB
        self.TimeConstantMutationMaxPower = neat_params.TimeConstantMutationMaxPower
        self.BiasMutationMaxPower = neat_params.BiasMutationMaxPower
        self.MutateNeuronTimeConstantsProb = neat_params.MutateNeuronTimeConstantsProb
        self.MutateNeuronBiasesProb = neat_params.MutateNeuronBiasesProb
        self.MinNeuronTimeConstant = neat_params.MinNeuronTimeConstant
        self.MaxNeuronTimeConstant = neat_params.MaxNeuronTimeConstant
        self.MinNeuronBias = neat_params.MinNeuronBias
        self.MaxNeuronBias = neat_params.MaxNeuronBias
        self.MutateNeuronActivationTypeProb = neat_params.MutateNeuronActivationTypeProb
        self.ActivationFunction_SignedSigmoid_Prob = neat_params.ActivationFunction_SignedSigmoid_Prob
        self.ActivationFunction_UnsignedSigmoid_Prob = neat_params.ActivationFunction_UnsignedSigmoid_Prob
        self.ActivationFunction_Tanh_Prob = neat_params.ActivationFunction_Tanh_Prob
        self.ActivationFunction_TanhCubic_Prob = neat_params.ActivationFunction_TanhCubic_Prob
        self.ActivationFunction_SignedStep_Prob = neat_params.ActivationFunction_SignedStep_Prob
        self.ActivationFunction_UnsignedStep_Prob = neat_params.ActivationFunction_UnsignedStep_Prob
        self.ActivationFunction_SignedGauss_Prob = neat_params.ActivationFunction_SignedGauss_Prob
        self.ActivationFunction_UnsignedGauss_Prob = neat_params.ActivationFunction_UnsignedGauss_Prob
        self.ActivationFunction_Abs_Prob = neat_params.ActivationFunction_Abs_Prob
        self.ActivationFunction_SignedSine_Prob = neat_params.ActivationFunction_SignedSine_Prob
        self.ActivationFunction_UnsignedSine_Prob = neat_params.ActivationFunction_UnsignedSine_Prob
        self.ActivationFunction_Linear_Prob = neat_params.ActivationFunction_Linear_Prob
        self.DontUseBiasNeuron = neat_params.DontUseBiasNeuron
        self.AllowLoops = neat_params.AllowLoops
        self.MutateNeuronTraitsProb = neat_params.MutateNeuronTraitsProb
        self.MutateLinkTraitsProb = neat_params.MutateLinkTraitsProb
        self.DisjointCoeff = neat_params.DisjointCoeff
        self.ExcessCoeff = neat_params.ExcessCoeff
        self.WeightDiffCoeff = neat_params.WeightDiffCoeff
        self.ActivationADiffCoeff = neat_params.ActivationADiffCoeff
        self.ActivationBDiffCoeff = neat_params.ActivationBDiffCoeff
        self.TimeConstantDiffCoeff = neat_params.TimeConstantDiffCoeff
        self.BiasDiffCoeff = neat_params.BiasDiffCoeff
        self.ActivationFunctionDiffCoeff = neat_params.ActivationFunctionDiffCoeff
        self.CompatTreshold = neat_params.CompatTreshold
        self.MinCompatTreshold = neat_params.MinCompatTreshold
        self.CompatTresholdModifier = neat_params.CompatTresholdModifier
        self.CompatTreshChangeInterval_Generations = neat_params.CompatTreshChangeInterval_Generations
        self.CompatTreshChangeInterval_Evaluations = neat_params.CompatTreshChangeInterval_Evaluations

        self.DivisionThreshold = neat_params.DivisionThreshold
        self.VarianceThreshold = neat_params.VarianceThreshold
        self.BandThreshold = neat_params.BandThreshold
        self.InitialDepth = neat_params.InitialDepth
        self.MaxDepth = neat_params.MaxDepth
        self.CPPN_Bias = neat_params.CPPN_Bias
        self.Width = neat_params.Width
        self.Height = neat_params.Height
        self.Qtree_Y = neat_params.Qtree_Y
        self.Qtree_X = neat_params.Qtree_X
        self.Leo = neat_params.Leo
        self.LeoThreshold = neat_params.LeoThreshold
        self.LeoSeed = neat_params.LeoSeed

        self.GeometrySeed = neat_params.GeometrySeed
        self.TournamentSize = neat_params.TournamentSize
        self.EliteFraction = neat_params.EliteFraction
