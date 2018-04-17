#include "Genome.h"
#include "Population.h"
#include "NeuralNetwork.h"
#include "Parameters.h"
#include "Substrate.h"
#include "Random.h"

#include <iostream>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace NEAT;

#define ENABLE_TESTING
#ifdef ENABLE_TESTING


int main() {
    Parameters params;

    params.PopulationSize = 25;
    params.DynamicCompatibility = true;
    params.AllowClones = false;
    params.CompatTreshold = 5.0;
    params.CompatTresholdModifier = 0.3;
    params.YoungAgeTreshold = 15;
    params.SpeciesMaxStagnation = 100;
    params.OldAgeTreshold = 35;
    params.MinSpecies = 2;
    params.MaxSpecies = 5;
    params.RouletteWheelSelection = true;
    params.RecurrentProb = 0.0;
    params.OverallMutationRate = 0.02;
    params.MutateWeightsProb = 0.90;
    params.WeightMutationMaxPower = 1.0;
    params.WeightReplacementMaxPower = 5.0;
    params.MutateWeightsSevereProb = 0.01;
    params.WeightMutationRate = 0.75;
    params.MaxWeight = 20;
    params.MutateAddNeuronProb = 0.01;
    params.MutateAddLinkProb = 0.02;
    params.MutateRemLinkProb = 0.00;
    params.SurvivalRate = 0.2;
    params.EliteFraction = 1;
    params.CrossoverRate = 0.5;
    params.MutateNeuronTraitsProb = 0;
    params.MutateLinkTraitsProb = 0;

    Genome g(0, 10, 0, 1, false, ActivationFunction::UNSIGNED_SIGMOID, ActivationFunction::UNSIGNED_SIGMOID, 0, params, 5);
    Population pop(g, params, true, 1.0, 0);


    return 0;
}

#endif
