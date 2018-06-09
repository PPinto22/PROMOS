#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include "Parameters.h"


namespace NEAT
{


// Load defaults
    void Parameters::Reset()
    {
        ////////////////////
        // Basic parameters
        ////////////////////

        // Size of population
        PopulationSize = 300;

        // If true, this enables dynamic compatibility thresholding
        // It will keep the number of species between MinSpecies and MaxSpecies
        DynamicCompatibility = true;

        // Minimum number of species
        MinSpecies = 5;

        // Maximum number of species
        MaxSpecies = 10;

        // Don't wipe the innovation database each generation?
        InnovationsForever = true;

        // Allow clones or nearly identical genomes to exist simultaneously in the population.
        // This is useful for non-deterministic environments,
        // as the same individual will get more than one chance to prove himself, also
        // there will be more chances the same individual to mutate in different ways.
        // The drawback is greatly increased time for reproduction. If you want to
        // search quickly, yet less efficient, leave this to true.
        AllowClones = true;

        // Keep an archive of genomes and don't allow any new genome to exist in the archive or the population
        ArchiveEnforcement = false;
    
        // When true, don't have a special bias neuron and treat all inputs equal
        DontUseBiasNeuron = true;
    
        // When false, this prevents any recurrent pathways in the genomes from forming
        AllowLoops = false;
    
        // Normalize genome size when calculating compatibility
        NormalizeGenomeSize = true;
    
        // Pointer to a function that specifies custom topology/trait constraints
        // Should return true if the genome FAILS to meet the constraints
        CustomConstraints = NULL;
        
        ////////////////////////////////
        // GA Parameters
        ////////////////////////////////

        // AgeGens threshold, meaning if a species is below it, it is considered young
        YoungAgeThreshold = 5;

        // Fitness boost multiplier for young species (1.0 means no boost)
        // Make sure it is >= 1.0 to avoid confusion
        YoungAgeFitnessBoost = 1.1;

        // Number of generations without improvement (stagnation) allowed for a species
        SpeciesMaxStagnation = 50;

        // Minimum jump in fitness necessary to be considered as improvement.
        // Setting this value to 0.0 makes the system to behave like regular NEAT.
        StagnationDelta = 0.0;

        // AgeGens threshold, meaning if a species is above it, it is considered old
        OldAgeThreshold = 30;

        // Multiplier that penalizes old species.
        // Make sure it is <= 1.0 to avoid confusion.
        OldAgePenalty = 0.5;

        // Detect competetive coevolution stagnation
        // This kills the worst species of age >N (each X generations)
        DetectCompetetiveCoevolutionStagnation = false;
        // Each X generation..
        KillWorstSpeciesEach = 15;
        // Of age above..
        KillWorstAge = 10;

        // Percent of best individuals that are allowed to reproduce. 1.0 = 100%
        SurvivalRate = 0.25;

        // Probability for a baby to result from sexual reproduction (crossover/mating). 1.0 = 100%
        // If asexual reprodiction is chosen, the baby will be mutated 100%
        CrossoverRate = 0.7;

        // If a baby results from sexual reproduction, this probability determines if mutation will
        // be performed after crossover. 1.0 = 100% (always mutate after crossover)
        OverallMutationRate = 0.25;

        // Probability for a baby to result from inter-species mating.
        InterspeciesCrossoverRate = 0.0001;

        // Probability for a baby to result from Multipoint Crossover when mating. 1.0 = 100%
        // The default is the Average mating.
        MultipointCrossoverRate = 0.75;

        // Performing roulette wheel selection or not?
        RouletteWheelSelection = false;

        // For tournament selection
        TournamentSize = 4;

        // Fraction of individuals to be copied unchanged
        EliteFraction = 0.01;





        ///////////////////////////////////
        // Phased Search parameters   //
        ///////////////////////////////////

        // Using phased search or not
        PhasedSearching = false;

        // Using delta coding or not
        DeltaCoding = false;

        // What is the MPC + base MPC needed to begin simplifying phase
        SimplifyingPhaseMPCThreshold = 20;

        // How many generations of global stagnation should have passed to enter simplifying phase
        SimplifyingPhaseStagnationThreshold = 30;

        // How many generations of MPC stagnation are needed to turn back on complexifying
        ComplexityFloorGenerations = 40;






        /////////////////////////////////////
        // Novelty Search parameters       //
        /////////////////////////////////////

        // the K constant
        NoveltySearch_K = 15;

        // Sparseness threshold. Add to the archive if above
        NoveltySearch_P_min = 0.5;

        // Dynamic Pmin?
        NoveltySearch_Dynamic_Pmin = true;

        // How many evaluations should pass without adding to the archive
        // in order to lower Pmin
        NoveltySearch_No_Archiving_Stagnation_Threshold = 150;

        // How should it be multiplied (make it less than 1.0)
        NoveltySearch_Pmin_lowering_multiplier = 0.9;

        // Not lower than this value
        NoveltySearch_Pmin_min = 0.05;

        // How many one-after-another additions to the archive should
        // pass in order to raise Pmin
        NoveltySearch_Quick_Archiving_Min_Evaluations = 8;

        // How should it be multiplied (make it more than 1.0)
        NoveltySearch_Pmin_raising_multiplier = 1.1;

        // Per how many evaluations to recompute the sparseness of the population
        NoveltySearch_Recompute_Sparseness_Each = 25;




        ///////////////////////////////////
        // Structural Mutation parameters
        ///////////////////////////////////

        // Probability for a baby to be mutated with the Add-Neuron mutation.
        MutateAddNeuronProb = 0.01;

        // Allow splitting of any recurrent links
        SplitRecurrent = true;

        // Allow splitting of looped recurrent links
        SplitLoopedRecurrent = true;

        // Probability for a baby to be mutated with the Add-Link mutation
        MutateAddLinkProb = 0.03;

        // Probability for a new incoming link to be from the bias neuron;
        // This enforces it. A value of 0.0 doesn't mean there will not be such links
        MutateAddLinkFromBiasProb = 0.0;

        // Probability for a baby to be mutated with the Remove-Link mutation
        MutateRemLinkProb = 0.0;

        // Probability for a baby that a simple neuron will be replaced with a link
        MutateRemSimpleNeuronProb = 0.0;

        // Maximum number of tries to find 2 neurons to add/remove a link
        LinkTries = 32;

        // Probability that a link mutation will be made recurrent
        RecurrentProb = 0;

        // Probability that a recurrent link mutation will be looped
        RecurrentLoopProb = 0;





        ///////////////////////////////////
        // Parameter Mutation parameters
        ///////////////////////////////////

        // Probability for a baby's weights to be mutated
        MutateWeightsProb = 0.90;

        // Probability for a severe (shaking) weight mutation
        MutateWeightsSevereProb = 0.25;

        // Probability for a particular gene's weight to be mutated. 1.0 = 100%
        WeightMutationRate = 1.0;

        // Maximum perturbation for a weight mutation
        WeightMutationMaxPower = 1.0;
    
        // Probability for a particular gene to be mutated via replacement of the weight. 1.0 = 100%
        WeightReplacementRate = 0.2;

        // Maximum magnitude of a replaced weight
        WeightReplacementMaxPower = 1.0;

        // Maximum absolute magnitude of a weight
        MaxWeight = 8.0;

        // Probability for a baby's A activation function parameters to be perturbed
        MutateActivationAProb = 0.0;

        // Probability for a baby's B activation function parameters to be perturbed
        MutateActivationBProb = 0.0;

        // Maximum magnitude for the A parameter perturbation
        ActivationAMutationMaxPower = 0.0;

        // Maximum magnitude for the B parameter perturbation
        ActivationBMutationMaxPower = 0.0;

        // Activation parameter A min/max
        MinActivationA = 1.0;
        MaxActivationA = 1.0;

        // Activation parameter B min/max
        MinActivationB = 0.0;
        MaxActivationB = 0.0;

        // Maximum magnitude for time costants perturbation
        TimeConstantMutationMaxPower = 0.0;

        // Maximum magnitude for biases perturbation
        BiasMutationMaxPower = WeightMutationMaxPower;

        // Probability for a baby's neuron time constant values to be mutated
        MutateNeuronTimeConstantsProb = 0.0;

        // Probability for a baby's neuron bias values to be mutated
        MutateNeuronBiasesProb = 0.0;

        // Time constant range
        MinNeuronTimeConstant = 0.0;
        MaxNeuronTimeConstant = 0.0;

        // Bias range
        MinNeuronBias = 0.0;
        MaxNeuronBias = 0.0;

        // Probability for a baby that an activation function type will be changed for a single neuron
        // considered a structural mutation because of the large impact on fitness
        MutateNeuronActivationTypeProb = 0.0;

        // Probabilities for a particular activation function appearance
        ActivationFunction_SignedSigmoid_Prob = 0.0;
        ActivationFunction_UnsignedSigmoid_Prob = 1.0;
        ActivationFunction_Tanh_Prob = 0.0;
        ActivationFunction_TanhCubic_Prob = 0.0;
        ActivationFunction_SignedStep_Prob = 0.0;
        ActivationFunction_UnsignedStep_Prob = 0.0;
        ActivationFunction_SignedGauss_Prob = 0.0;
        ActivationFunction_UnsignedGauss_Prob = 0.0;
        ActivationFunction_Abs_Prob = 0.0;
        ActivationFunction_SignedSine_Prob = 0.0;
        ActivationFunction_UnsignedSine_Prob = 0.0;
        ActivationFunction_Linear_Prob = 0.0;
        ActivationFunction_Relu_Prob = 0.0;
        ActivationFunction_Softplus_Prob = 0.0;


        // Trait mutation probabilities
        MutateNeuronTraitsProb = 1.0;
        MutateLinkTraitsProb = 1.0;
        MutateGenomeTraitsProb = 1.0;


        /////////////////////////////
        // Genome properties params
        /////////////////////////////

    
        /////////////////////////////////////
        // Speciation parameters
        /////////////////////////////////////

        // Percent of disjoint genes importance
        DisjointCoeff = 1.0;

        // Percent of excess genes importance
        ExcessCoeff = 1.0;

        // Average weight difference importance
        WeightDiffCoeff = 0.5;

        // Node-specific activation parameter A difference importance
        ActivationADiffCoeff = 0.1;

        // Node-specific activation parameter B difference importance
        ActivationBDiffCoeff = 0.1;

        // Average time constant difference importance
        TimeConstantDiffCoeff = 0.1;

        // Average bias difference importance
        BiasDiffCoeff = 0.2;

        // Activation function type difference importance
        ActivationFunctionDiffCoeff = 0.2;

        // Compatibility threshold
        CompatThreshold = 5.0;

        // Minumal value of the compatibility threshold
        MinCompatThreshold = 0.2;

        // Modifier per generation for keeping the species stable
        CompatThresholdModifier = 0.3;

        // Per how many generations to change the threshold
        // (used in generational mode)
        CompatThreshChangeInterval_Generations = 1;

        // Per how many evaluations to change the threshold
        // (used in steady state mode)
        CompatThreshChangeInterval_Evaluations = 10;




        //////////////////////////////
        // ES-HyperNEAT parameters

        DivisionThreshold = 0.03;

        VarianceThreshold = 0.03;

        // Used for Band prunning.
        BandThreshold = 0.3;

        // Max and Min Depths of the quadtree
        InitialDepth = 3;
        MaxDepth = 3;

        // How many hidden layers before connecting nodes to output. At 0 there is
        // one hidden layer. At 1, there are two and so on.
        IterationLevel = 1;

        // The Bias value for the CPPN queries.
        CPPN_Bias = 1.0;

        // Quadtree Dimensions
        // The range of the tree. Typically set to 2,
        Width = 2.0;

        Height = 2.0;

        // The (x, y) coordinates of the tree
        Qtree_X = 0.0;

        Qtree_Y = 0.0;

        // Use Link Expression output
        Leo = false;

        // Threshold above which a connection is expressed
        LeoThreshold = 0.1;

        // Use geometric seeding. Currently only along the X axis. 1
        LeoSeed = false;

        GeometrySeed = false;
    }

    Parameters::Parameters()
    {
        Reset();
    }

    int Parameters::Load(std::ifstream &a_DataFile)
    {
        std::string s, tf;
        do
        {
        	if(a_DataFile.eof()){
        		throw std::invalid_argument( "Reached end of file before reading the start tag. "
        			"A valid parameters file must start with NEAT_ParametersStart and end with NEAT_ParametersEnd." );
        	}
            a_DataFile >> s;
        }
        while (s != "NEAT_ParametersStart");

        a_DataFile >> s;
        while (s != "NEAT_ParametersEnd")
        {
			if(a_DataFile.eof()){
        		throw std::invalid_argument( "Reached end of file before reading the end tag. "
        			"A valid parameters file must start with NEAT_ParametersStart and end with NEAT_ParametersEnd." );
        	}

            if (s == "PopulationSize")
                a_DataFile >> PopulationSize;

            else if (s == "DynamicCompatibility")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    DynamicCompatibility = true;
                else
                    DynamicCompatibility = false;
            }

            else if (s == "MinSpecies")
                a_DataFile >> MinSpecies;

            else if (s == "MaxSpecies")
                a_DataFile >> MaxSpecies;

            else if (s == "InnovationsForever")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    InnovationsForever = true;
                else
                    InnovationsForever = false;
            }

            else if (s == "AllowClones")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    AllowClones = true;
                else
                    AllowClones = false;
            }

            else if (s == "NormalizeGenomeSize")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    NormalizeGenomeSize = true;
                else
                    NormalizeGenomeSize = false;
            }


            else if (s == "YoungAgeThreshold")
                a_DataFile >> YoungAgeThreshold;

            else if (s == "YoungAgeFitnessBoost")
                a_DataFile >> YoungAgeFitnessBoost;

            else if (s == "SpeciesDropoffAge")
                a_DataFile >> SpeciesMaxStagnation;

            else if (s == "StagnationDelta")
                a_DataFile >> StagnationDelta;

            else if (s == "OldAgeThreshold")
                a_DataFile >> OldAgeThreshold;

            else if (s == "OldAgePenalty")
                a_DataFile >> OldAgePenalty;

            else if (s == "DetectCompetetiveCoevolutionStagnation")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    DetectCompetetiveCoevolutionStagnation = true;
                else
                    DetectCompetetiveCoevolutionStagnation = false;
            }

            else if (s == "KillWorstSpeciesEach")
                a_DataFile >> KillWorstSpeciesEach;

            else if (s == "KillWorstAge")
                a_DataFile >> KillWorstAge;

            else if (s == "SurvivalRate")
                a_DataFile >> SurvivalRate;

            else if (s == "CrossoverRate")
                a_DataFile >> CrossoverRate;

            else if (s == "OverallMutationRate")
                a_DataFile >> OverallMutationRate;

            else if (s == "InterspeciesCrossoverRate")
                a_DataFile >> InterspeciesCrossoverRate;

            else if (s == "MultipointCrossoverRate")
                a_DataFile >> MultipointCrossoverRate;

            else if (s == "RouletteWheelSelection")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    RouletteWheelSelection = true;
                else
                    RouletteWheelSelection = false;
            }

            else if (s == "PhasedSearching")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    PhasedSearching = true;
                else
                    PhasedSearching = false;
            }

            else if (s == "DeltaCoding")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    DeltaCoding = true;
                else
                    DeltaCoding = false;
            }

            else if (s == "SimplifyingPhaseMPCThreshold")
                a_DataFile >> SimplifyingPhaseMPCThreshold;

            else if (s == "SimplifyingPhaseStagnationThreshold")
                a_DataFile >> SimplifyingPhaseStagnationThreshold;

            else if (s == "ComplexityFloorGenerations")
                a_DataFile >> ComplexityFloorGenerations;

            else if (s == "NoveltySearch_K")
                a_DataFile >> NoveltySearch_K;

            else if (s == "NoveltySearch_P_min")
                a_DataFile >> NoveltySearch_P_min;

            else if (s == "NoveltySearch_Dynamic_Pmin")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    NoveltySearch_Dynamic_Pmin = true;
                else
                    NoveltySearch_Dynamic_Pmin = false;
            }

            else if (s == "NoveltySearch_No_Archiving_Stagnation_Threshold")
                a_DataFile >> NoveltySearch_No_Archiving_Stagnation_Threshold;

            else if (s == "NoveltySearch_Pmin_lowering_multiplier")
                a_DataFile >> NoveltySearch_Pmin_lowering_multiplier;

            else if (s == "NoveltySearch_Pmin_min")
                a_DataFile >> NoveltySearch_Pmin_min;

            else if (s == "NoveltySearch_Quick_Archiving_Min_Evaluations")
                a_DataFile >> NoveltySearch_Quick_Archiving_Min_Evaluations;

            else if (s == "NoveltySearch_Pmin_raising_multiplier")
                a_DataFile >> NoveltySearch_Pmin_raising_multiplier;

            else if (s == "NoveltySearch_Recompute_Sparseness_Each")
                a_DataFile >> NoveltySearch_Recompute_Sparseness_Each;

            else if (s == "MutateAddNeuronProb")
                a_DataFile >> MutateAddNeuronProb;

            else if (s == "SplitRecurrent")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    SplitRecurrent = true;
                else
                    SplitRecurrent = false;
            }

            else if (s == "SplitLoopedRecurrent")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    SplitLoopedRecurrent = true;
                else
                    SplitLoopedRecurrent = false;
            }

            else if (s == "MutateAddLinkProb")
                a_DataFile >> MutateAddLinkProb;

            else if (s == "MutateAddLinkFromBiasProb")
                a_DataFile >> MutateAddLinkFromBiasProb;

            else if (s == "MutateRemLinkProb")
                a_DataFile >> MutateRemLinkProb;

            else if (s == "MutateRemSimpleNeuronProb")
                a_DataFile >> MutateRemSimpleNeuronProb;

            else if (s == "LinkTries")
                a_DataFile >> LinkTries;

            else if (s == "RecurrentProb")
                a_DataFile >> RecurrentProb;

            else if (s == "RecurrentLoopProb")
                a_DataFile >> RecurrentLoopProb;

            else if (s == "MutateWeightsProb")
                a_DataFile >> MutateWeightsProb;

            else if (s == "MutateWeightsSevereProb")
                a_DataFile >> MutateWeightsSevereProb;

            else if (s == "WeightMutationRate")
                a_DataFile >> WeightMutationRate;

            else if (s == "WeightMutationMaxPower")
                a_DataFile >> WeightMutationMaxPower;

            else if (s == "WeightReplacementRate")
                a_DataFile >> WeightReplacementRate;

            else if (s == "WeightReplacementMaxPower")
                a_DataFile >> WeightReplacementMaxPower;

            else if (s == "MaxWeight")
                a_DataFile >> MaxWeight;

            else if (s == "MutateActivationAProb")
                a_DataFile >> MutateActivationAProb;

            else if (s == "MutateActivationBProb")
                a_DataFile >> MutateActivationBProb;

            else if (s == "ActivationAMutationMaxPower")
                a_DataFile >> ActivationAMutationMaxPower;

            else if (s == "ActivationBMutationMaxPower")
                a_DataFile >> ActivationBMutationMaxPower;

            else if (s == "MinActivationA")
                a_DataFile >> MinActivationA;

            else if (s == "MaxActivationA")
                a_DataFile >> MaxActivationA;

            else if (s == "MinActivationB")
                a_DataFile >> MinActivationB;

            else if (s == "MaxActivationB")
                a_DataFile >> MaxActivationB;

            else if (s == "TimeConstantMutationMaxPower")
                a_DataFile >> TimeConstantMutationMaxPower;

            else if (s == "BiasMutationMaxPower")
                a_DataFile >> BiasMutationMaxPower;

            else if (s == "MutateNeuronTimeConstantsProb")
                a_DataFile >> MutateNeuronTimeConstantsProb;

            else if (s == "MutateNeuronBiasesProb")
                a_DataFile >> MutateNeuronBiasesProb;

            else if (s == "MinNeuronTimeConstant")
                a_DataFile >> MinNeuronTimeConstant;

            else if (s == "MaxNeuronTimeConstant")
                a_DataFile >> MaxNeuronTimeConstant;

            else if (s == "MinNeuronBias")
                a_DataFile >> MinNeuronBias;

            else if (s == "MaxNeuronBias")
                a_DataFile >> MaxNeuronBias;

            else if (s == "MutateNeuronActivationTypeProb")
                a_DataFile >> MutateNeuronActivationTypeProb;

            else if (s == "ActivationFunction_SignedSigmoid_Prob")
                a_DataFile >> ActivationFunction_SignedSigmoid_Prob;
            else if (s == "ActivationFunction_UnsignedSigmoid_Prob")
                a_DataFile >> ActivationFunction_UnsignedSigmoid_Prob;
            else if (s == "ActivationFunction_Tanh_Prob")
                a_DataFile >> ActivationFunction_Tanh_Prob;
            else if (s == "ActivationFunction_TanhCubic_Prob")
                a_DataFile >> ActivationFunction_TanhCubic_Prob;
            else if (s == "ActivationFunction_SignedStep_Prob")
                a_DataFile >> ActivationFunction_SignedStep_Prob;
            else if (s == "ActivationFunction_UnsignedStep_Prob")
                a_DataFile >> ActivationFunction_UnsignedStep_Prob;
            else if (s == "ActivationFunction_SignedGauss_Prob")
                a_DataFile >> ActivationFunction_SignedGauss_Prob;
            else if (s == "ActivationFunction_UnsignedGauss_Prob")
                a_DataFile >> ActivationFunction_UnsignedGauss_Prob;
            else if (s == "ActivationFunction_Abs_Prob")
                a_DataFile >> ActivationFunction_Abs_Prob;
            else if (s == "ActivationFunction_SignedSine_Prob")
                a_DataFile >> ActivationFunction_SignedSine_Prob;
            else if (s == "ActivationFunction_UnsignedSine_Prob")
                a_DataFile >> ActivationFunction_UnsignedSine_Prob;
            else if (s == "ActivationFunction_Linear_Prob")
                a_DataFile >> ActivationFunction_Linear_Prob;
            else if (s == "ActivationFunction_Relu_Prob")
                a_DataFile >> ActivationFunction_Relu_Prob;
            else if (s == "ActivationFunction_Softplus_Prob")
                a_DataFile >> ActivationFunction_Softplus_Prob;

            else if (s == "DontUseBiasNeuron")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    DontUseBiasNeuron = true;
                else
                    DontUseBiasNeuron = false;
            }

            else if (s == "AllowLoops")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    AllowLoops = true;
                else
                    AllowLoops = false;
            }

            else if (s == "ArchiveEnforcement")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    ArchiveEnforcement = true;
                else
                    ArchiveEnforcement = false;
            }

            else if (s == "DisjointCoeff")
                a_DataFile >> DisjointCoeff;

            else if (s == "ExcessCoeff")
                a_DataFile >> ExcessCoeff;

            else if (s == "WeightDiffCoeff")
                a_DataFile >> WeightDiffCoeff;

            else if (s == "ActivationADiffCoeff")
                a_DataFile >> ActivationADiffCoeff;

            else if (s == "ActivationBDiffCoeff")
                a_DataFile >> ActivationBDiffCoeff;

            else if (s == "TimeConstantDiffCoeff")
                a_DataFile >> TimeConstantDiffCoeff;

            else if (s == "BiasDiffCoeff")
                a_DataFile >> BiasDiffCoeff;

            else if (s == "ActivationFunctionDiffCoeff")
                a_DataFile >> ActivationFunctionDiffCoeff;

            else if (s == "CompatThreshold")
                a_DataFile >> CompatThreshold;

            else if (s == "MinCompatThreshold")
                a_DataFile >> MinCompatThreshold;

            else if (s == "CompatThresholdModifier")
                a_DataFile >> CompatThresholdModifier;

            else if (s == "CompatThreshChangeInterval_Generations")
                a_DataFile >> CompatThreshChangeInterval_Generations;

            else if (s == "CompatThreshChangeInterval_Evaluations")
                a_DataFile >> CompatThreshChangeInterval_Evaluations;

            else if (s == "DivisionThreshold")
                a_DataFile >> DivisionThreshold;

            else if (s == "VarianceThreshold")
                a_DataFile >> VarianceThreshold;

            else if (s == "BandThreshold")
                a_DataFile >> BandThreshold;

            else if (s == "InitialDepth")
                a_DataFile >> InitialDepth;

            else if (s == "MaxDepth")
                a_DataFile >> MaxDepth;

            else if (s == "IterationLevel")
                a_DataFile >> IterationLevel;

            else if (s == "TournamentSize")
                a_DataFile >> TournamentSize;

            else if (s == "CPPN_Bias")
                a_DataFile >> CPPN_Bias;

            else if (s == "Width")
                a_DataFile >> Width;

            else if (s == "Height")
                a_DataFile >> Height;

            else if (s == "Qtree_X")
                a_DataFile >> Qtree_X;

            else if (s == "Qtree_Y")
                a_DataFile >> Qtree_Y;

            else if (s == "Leo")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    Leo = true;
                else
                    Leo = false;
            }
            else if (s == "GeometrySeed")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    GeometrySeed = true;
                else
                    GeometrySeed = false;
            }

            else if (s == "LeoThreshold")
                a_DataFile >> LeoThreshold;

            else if (s == "LeoSeed")
            {
                a_DataFile >> tf;
                if (tf == "true" || tf == "1" || tf == "1.0")
                    LeoSeed = true;
                else
                    LeoSeed = false;
            }
            else if (s == "Elitism")
            {
                a_DataFile >> EliteFraction;
            }

            else if (s == "MutateNeuronTraitsProb"){
            	a_DataFile >> MutateNeuronTraitsProb;
            }
            else if (s == "MutateLinkTraitsProb"){
            	a_DataFile >> MutateLinkTraitsProb;
            }
            else if (s == "MutateGenomeTraitsProb"){
            	a_DataFile >> MutateGenomeTraitsProb;
            }
            else{
                throw std::invalid_argument( std::string("Unrecognized parameter: ") + s );
            }

            // End of else if chain
            if(!a_DataFile){
            	throw std::invalid_argument( "Error while parsing parameters file." );
            }
            
            a_DataFile >> s;
        }

        return 0;
    }


    int Parameters::Load(const char *a_FileName)
    {
        std::ifstream data(a_FileName);
        if (!data.is_open())
            throw std::invalid_argument( std::string("Could not open ") + a_FileName );

        int result = Load(data);
        data.close();
        return result;
    }

    void Parameters::Save(const char *filename)
    {
        FILE *f = fopen(filename, "w");
        Save(f);
        fclose(f);
    }


    void Parameters::Save(FILE *a_fstream)
    {
        fprintf(a_fstream, "NEAT_ParametersStart\n");

        fprintf(a_fstream, "PopulationSize %d\n", PopulationSize);
        fprintf(a_fstream, "DynamicCompatibility %s\n", DynamicCompatibility == true ? "true" : "false");
        fprintf(a_fstream, "MinSpecies %d\n", MinSpecies);
        fprintf(a_fstream, "MaxSpecies %d\n", MaxSpecies);
        fprintf(a_fstream, "InnovationsForever %s\n", InnovationsForever == true ? "true" : "false");
        fprintf(a_fstream, "AllowClones %s\n", AllowClones == true ? "true" : "false");
        fprintf(a_fstream, "NormalizeGenomeSize %s\n", NormalizeGenomeSize == true ? "true" : "false");
        fprintf(a_fstream, "YoungAgeThreshold %d\n", YoungAgeThreshold);
        fprintf(a_fstream, "YoungAgeFitnessBoost %3.20f\n", YoungAgeFitnessBoost);
        fprintf(a_fstream, "SpeciesDropoffAge %d\n", SpeciesMaxStagnation);
        fprintf(a_fstream, "StagnationDelta %3.20f\n", StagnationDelta);
        fprintf(a_fstream, "OldAgeThreshold %d\n", OldAgeThreshold);
        fprintf(a_fstream, "OldAgePenalty %3.20f\n", OldAgePenalty);
        fprintf(a_fstream, "DetectCompetetiveCoevolutionStagnation %s\n",
                DetectCompetetiveCoevolutionStagnation == true ? "true" : "false");
        fprintf(a_fstream, "KillWorstSpeciesEach %d\n", KillWorstSpeciesEach);
        fprintf(a_fstream, "KillWorstAge %d\n", KillWorstAge);
        fprintf(a_fstream, "SurvivalRate %3.20f\n", SurvivalRate);
        fprintf(a_fstream, "CrossoverRate %3.20f\n", CrossoverRate);
        fprintf(a_fstream, "OverallMutationRate %3.20f\n", OverallMutationRate);
        fprintf(a_fstream, "InterspeciesCrossoverRate %3.20f\n", InterspeciesCrossoverRate);
        fprintf(a_fstream, "MultipointCrossoverRate %3.20f\n", MultipointCrossoverRate);
        fprintf(a_fstream, "RouletteWheelSelection %s\n", RouletteWheelSelection == true ? "true" : "false");
        fprintf(a_fstream, "PhasedSearching %s\n", PhasedSearching == true ? "true" : "false");
        fprintf(a_fstream, "DeltaCoding %s\n", DeltaCoding == true ? "true" : "false");
        fprintf(a_fstream, "SimplifyingPhaseMPCThreshold %d\n", SimplifyingPhaseMPCThreshold);
        fprintf(a_fstream, "SimplifyingPhaseStagnationThreshold %d\n", SimplifyingPhaseStagnationThreshold);
        fprintf(a_fstream, "ComplexityFloorGenerations %d\n", ComplexityFloorGenerations);
        fprintf(a_fstream, "NoveltySearch_K %d\n", NoveltySearch_K);
        fprintf(a_fstream, "NoveltySearch_P_min %3.20f\n", NoveltySearch_P_min);
        fprintf(a_fstream, "NoveltySearch_Dynamic_Pmin %s\n", NoveltySearch_Dynamic_Pmin == true ? "true" : "false");
        fprintf(a_fstream, "NoveltySearch_No_Archiving_Stagnation_Threshold %d\n",
                NoveltySearch_No_Archiving_Stagnation_Threshold);
        fprintf(a_fstream, "NoveltySearch_Pmin_lowering_multiplier %3.20f\n", NoveltySearch_Pmin_lowering_multiplier);
        fprintf(a_fstream, "NoveltySearch_Pmin_min %3.20f\n", NoveltySearch_Pmin_min);
        fprintf(a_fstream, "NoveltySearch_Quick_Archiving_Min_Evaluations %d\n",
                NoveltySearch_Quick_Archiving_Min_Evaluations);
        fprintf(a_fstream, "NoveltySearch_Pmin_raising_multiplier %3.20f\n", NoveltySearch_Pmin_raising_multiplier);
        fprintf(a_fstream, "NoveltySearch_Recompute_Sparseness_Each %d\n", NoveltySearch_Recompute_Sparseness_Each);
        fprintf(a_fstream, "MutateAddNeuronProb %3.20f\n", MutateAddNeuronProb);
        fprintf(a_fstream, "SplitRecurrent %s\n", SplitRecurrent == true ? "true" : "false");
        fprintf(a_fstream, "SplitLoopedRecurrent %s\n", SplitLoopedRecurrent == true ? "true" : "false");
        fprintf(a_fstream, "NeuronTries %d\n", NeuronTries);
        fprintf(a_fstream, "MutateAddLinkProb %3.20f\n", MutateAddLinkProb);
        fprintf(a_fstream, "MutateAddLinkFromBiasProb %3.20f\n", MutateAddLinkFromBiasProb);
        fprintf(a_fstream, "MutateRemLinkProb %3.20f\n", MutateRemLinkProb);
        fprintf(a_fstream, "MutateRemSimpleNeuronProb %3.20f\n", MutateRemSimpleNeuronProb);
        fprintf(a_fstream, "LinkTries %d\n", LinkTries);
        fprintf(a_fstream, "RecurrentProb %3.20f\n", RecurrentProb);
        fprintf(a_fstream, "RecurrentLoopProb %3.20f\n", RecurrentLoopProb);
        fprintf(a_fstream, "MutateWeightsProb %3.20f\n", MutateWeightsProb);
        fprintf(a_fstream, "MutateWeightsSevereProb %3.20f\n", MutateWeightsSevereProb);
        fprintf(a_fstream, "WeightMutationRate %3.20f\n", WeightMutationRate);
        fprintf(a_fstream, "WeightMutationMaxPower %3.20f\n", WeightMutationMaxPower);
        fprintf(a_fstream, "WeightReplacementRate %3.20f\n", WeightReplacementRate);
        fprintf(a_fstream, "WeightReplacementMaxPower %3.20f\n", WeightReplacementMaxPower);
        fprintf(a_fstream, "MaxWeight %3.20f\n", MaxWeight);
        fprintf(a_fstream, "MutateActivationAProb %3.20f\n", MutateActivationAProb);
        fprintf(a_fstream, "MutateActivationBProb %3.20f\n", MutateActivationBProb);
        fprintf(a_fstream, "ActivationAMutationMaxPower %3.20f\n", ActivationAMutationMaxPower);
        fprintf(a_fstream, "ActivationBMutationMaxPower %3.20f\n", ActivationBMutationMaxPower);
        fprintf(a_fstream, "TimeConstantMutationMaxPower %3.20f\n", TimeConstantMutationMaxPower);
        fprintf(a_fstream, "BiasMutationMaxPower %3.20f\n", BiasMutationMaxPower);
        fprintf(a_fstream, "MinActivationA %3.20f\n", MinActivationA);
        fprintf(a_fstream, "MaxActivationA %3.20f\n", MaxActivationA);
        fprintf(a_fstream, "MinActivationB %3.20f\n", MinActivationB);
        fprintf(a_fstream, "MaxActivationB %3.20f\n", MaxActivationB);
        fprintf(a_fstream, "MutateNeuronActivationTypeProb %3.20f\n", MutateNeuronActivationTypeProb);
        fprintf(a_fstream, "ActivationFunction_SignedSigmoid_Prob %3.20f\n", ActivationFunction_SignedSigmoid_Prob);
        fprintf(a_fstream, "ActivationFunction_UnsignedSigmoid_Prob %3.20f\n", ActivationFunction_UnsignedSigmoid_Prob);
        fprintf(a_fstream, "ActivationFunction_Tanh_Prob %3.20f\n", ActivationFunction_Tanh_Prob);
        fprintf(a_fstream, "ActivationFunction_TanhCubic_Prob %3.20f\n", ActivationFunction_TanhCubic_Prob);
        fprintf(a_fstream, "ActivationFunction_SignedStep_Prob %3.20f\n", ActivationFunction_SignedStep_Prob);
        fprintf(a_fstream, "ActivationFunction_UnsignedStep_Prob %3.20f\n", ActivationFunction_UnsignedStep_Prob);
        fprintf(a_fstream, "ActivationFunction_SignedGauss_Prob %3.20f\n", ActivationFunction_SignedGauss_Prob);
        fprintf(a_fstream, "ActivationFunction_UnsignedGauss_Prob %3.20f\n", ActivationFunction_UnsignedGauss_Prob);
        fprintf(a_fstream, "ActivationFunction_Abs_Prob %3.20f\n", ActivationFunction_Abs_Prob);
        fprintf(a_fstream, "ActivationFunction_SignedSine_Prob %3.20f\n", ActivationFunction_SignedSine_Prob);
        fprintf(a_fstream, "ActivationFunction_UnsignedSine_Prob %3.20f\n", ActivationFunction_UnsignedSine_Prob);
        fprintf(a_fstream, "ActivationFunction_Linear_Prob %3.20f\n", ActivationFunction_Linear_Prob);
        fprintf(a_fstream, "ActivationFunction_Relu_Prob %3.20f\n", ActivationFunction_Relu_Prob);
        fprintf(a_fstream, "ActivationFunction_Softplus_Prob %3.20f\n", ActivationFunction_Softplus_Prob);
        fprintf(a_fstream, "MutateNeuronTimeConstantsProb %3.20f\n", MutateNeuronTimeConstantsProb);
        fprintf(a_fstream, "MutateNeuronBiasesProb %3.20f\n", MutateNeuronBiasesProb);
        fprintf(a_fstream, "MinNeuronTimeConstant %3.20f\n", MinNeuronTimeConstant);
        fprintf(a_fstream, "MaxNeuronTimeConstant %3.20f\n", MaxNeuronTimeConstant);
        fprintf(a_fstream, "MinNeuronBias %3.20f\n", MinNeuronBias);
        fprintf(a_fstream, "MaxNeuronBias %3.20f\n", MaxNeuronBias);
        fprintf(a_fstream, "DontUseBiasNeuron %s\n", DontUseBiasNeuron == true ? "true" : "false");
        fprintf(a_fstream, "ArchiveEnforcement %s\n", ArchiveEnforcement == true ? "true" : "false");
        fprintf(a_fstream, "AllowLoops %s\n", AllowLoops == true ? "true" : "false");
        fprintf(a_fstream, "DisjointCoeff %3.20f\n", DisjointCoeff);
        fprintf(a_fstream, "ExcessCoeff %3.20f\n", ExcessCoeff);
        fprintf(a_fstream, "ActivationADiffCoeff %3.20f\n", ActivationADiffCoeff);
        fprintf(a_fstream, "ActivationBDiffCoeff %3.20f\n", ActivationBDiffCoeff);
        fprintf(a_fstream, "WeightDiffCoeff %3.20f\n", WeightDiffCoeff);
        fprintf(a_fstream, "TimeConstantDiffCoeff %3.20f\n", TimeConstantDiffCoeff);
        fprintf(a_fstream, "BiasDiffCoeff %3.20f\n", BiasDiffCoeff);
        fprintf(a_fstream, "ActivationFunctionDiffCoeff %3.20f\n", ActivationFunctionDiffCoeff);
        fprintf(a_fstream, "CompatThreshold %3.20f\n", CompatThreshold);
        fprintf(a_fstream, "MinCompatThreshold %3.20f\n", MinCompatThreshold);
        fprintf(a_fstream, "CompatThresholdModifier %3.20f\n", CompatThresholdModifier);
        fprintf(a_fstream, "CompatThreshChangeInterval_Generations %d\n", CompatThreshChangeInterval_Generations);
        fprintf(a_fstream, "CompatThreshChangeInterval_Evaluations %d\n", CompatThreshChangeInterval_Evaluations);


        fprintf(a_fstream, "DivisionThreshold %3.20f\n", DivisionThreshold);
        fprintf(a_fstream, "VarianceThreshold %3.20f\n", VarianceThreshold);
        fprintf(a_fstream, "BandThreshold %3.20f\n", BandThreshold);
        fprintf(a_fstream, "InitialDepth %d\n", InitialDepth);
        fprintf(a_fstream, "MaxDepth %d\n", MaxDepth);
        fprintf(a_fstream, "IterationLevel %d\n", IterationLevel);
        fprintf(a_fstream, "TournamentSize %d\n", TournamentSize);
        fprintf(a_fstream, "CPPN_Bias %3.20f\n", CPPN_Bias);
        fprintf(a_fstream, "Width %3.20f\n", Width);
        fprintf(a_fstream, "Height %3.20f\n", Height);
        fprintf(a_fstream, "Qtree_X %3.20f\n", Qtree_X);
        fprintf(a_fstream, "Qtree_Y %3.20f\n", Qtree_Y);
        fprintf(a_fstream, "Leo %s\n", Leo == true ? "true" : "false");
        fprintf(a_fstream, "LeoThreshold %3.20f\n", LeoThreshold);
        fprintf(a_fstream, "LeoSeed %s\n", LeoSeed == true ? "true" : "false");
        fprintf(a_fstream, "GeometrySeed %s\n", GeometrySeed == true ? "true" : "false");
        fprintf(a_fstream, "Elitism %3.20f\n", EliteFraction);

        fprintf(a_fstream, "NEAT_ParametersEnd\n");
    }


} // namespace NEAT
