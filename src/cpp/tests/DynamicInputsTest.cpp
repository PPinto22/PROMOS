#include "../MultiNEAT/Genome.h"
#include "../MultiNEAT/Population.h"
#include "../MultiNEAT/NeuralNetwork.h"
#include "../MultiNEAT/Parameters.h"
#include "../MultiNEAT/Random.h"

using namespace NEAT;
using namespace std;

int main() {
    Parameters params;
    params.PopulationSize = 25;
    params.DynamicCompatibility = true;
    params.DontUseBiasNeuron = false; // !
    params.AllowClones = false;
    params.CompatThreshold = 5.0;
    params.CompatThresholdModifier = 0.3;
    params.YoungAgeThreshold = 15;
    params.SpeciesMaxStagnation = 100;
    params.OldAgeThreshold = 35;
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
    params.MutateAddNeuronProb = 1.0; // !
    params.MutateAddLinkProb = 0.02;
    params.MutateRemLinkProb = 0.00;
    params.SurvivalRate = 0.2;
    params.EliteFraction = 0.1;
    params.CrossoverRate = 0.5;
    params.MutateNeuronTraitsProb = 0;
    params.MutateLinkTraitsProb = 0;

    Genome g(0, 10, 2, 1, false, ActivationFunction::UNSIGNED_SIGMOID, ActivationFunction::UNSIGNED_SIGMOID, 1, params, 1);
    Population pop(g, params, true, 1.0, 0);
//    pop.Epoch();
    pop.ResizeInputs(20);

    cout << "Finish" << endl;
  }
