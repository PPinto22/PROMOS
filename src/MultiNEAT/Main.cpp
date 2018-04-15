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

    params.PopulationSize = 10;
    params.DynamicCompatibility = true;
    params.NormalizeGenomeSize = true;
    params.WeightDiffCoeff = 0.1;
    params.CompatTreshold = 2.0;
    params.YoungAgeTreshold = 15;
    params.SpeciesMaxStagnation = 15;
    params.OldAgeTreshold = 35;
    params.MinSpecies = 2;
    params.MaxSpecies = 10;
    params.RouletteWheelSelection = false;
    params.RecurrentProb = 0.0;
    params.OverallMutationRate = 1.0;

    params.ArchiveEnforcement = false;

    params.MutateWeightsProb = 0.05;

    params.WeightMutationMaxPower = 0.5;
    params.WeightReplacementMaxPower = 8.0;
    params.MutateWeightsSevereProb = 0.0;
    params.WeightMutationRate = 0.25;
    params.WeightReplacementRate = 0.9;

    params.MaxWeight = 8;

    params.MutateAddNeuronProb = 0.001;
    params.MutateAddLinkProb = 0.3;
    params.MutateRemLinkProb = 0.0;

    params.MinActivationA = 4.9;
    params.MaxActivationA = 4.9;

    params.ActivationFunction_SignedSigmoid_Prob = 0.0;
    params.ActivationFunction_UnsignedSigmoid_Prob = 1.0;
    params.ActivationFunction_Tanh_Prob = 0.0;
    params.ActivationFunction_SignedStep_Prob = 0.0;

    params.CrossoverRate = 0.0;
    params.MultipointCrossoverRate = 0.0;
    params.SurvivalRate = 0.2;

    params.MutateNeuronTraitsProb = 0;
    params.MutateLinkTraitsProb = 0;

    params.AllowLoops = true;
    params.AllowClones = true;


    Genome s(0, 1, 1, 1, false, UNSIGNED_SIGMOID, UNSIGNED_SIGMOID, 0, params, 2);

    Population pop(s, params, true, 1.0, 0);
    Population pop2(s, params, true, 1.0, 0);

    RNG rng;
    rng.Seed(0);

    for (int k = 0; k < 5000; k++) {
        double bestf = -999999;
        for (unsigned int i = 0; i < pop.m_Species.size(); i++) {
            for (unsigned int j = 0; j < pop.m_Species[i].m_Individuals.size(); j++) {
                double f = rng.RandFloat();
                pop.m_Species[i].m_Individuals[j].SetFitness(rng.RandFloat());
                pop.m_Species[i].m_Individuals[j].SetEvaluated();


                if (f > bestf) {
                    bestf = f;
                }
            }
        }

        Genome g = pop.GetBestGenome();

        printf("Generation: %d, best fitness: %3.5f\n", k, bestf);
        printf("Species: %ld\n", pop.m_Species.size());
        pop.Epoch();

        if(pop.m_Species.size() > 1){
            printf("Species: %ld\n", pop.m_Species.size());
            cout << "[DEBUG] 1\n";
            std::vector<Genome> genomes = pop.GetBestGenomesBySpecies(4);
            cout << "[DEBUG] 2\n";
            pop.ReplaceGenomes(genomes);
            cout << "[DEBUG] 3\n";
        }
    }

    return 0;
}

#endif
