#ifndef POPULATION_H
#define POPULATION_H

#include <vector>
#include <float.h>

#include "Innovation.h"
#include "Genome.h"
#include "PhenotypeBehavior.h"
#include "Genes.h"
#include "Species.h"
#include "Parameters.h"
#include "Random.h"

#ifdef USE_BOOST_PYTHON
#include <boost/python.hpp>
namespace py = boost::python;
#endif

namespace NEAT
{

//////////////////////////////////////////////
// The Population class
//////////////////////////////////////////////

enum SearchMode
{
    COMPLEXIFYING,
    SIMPLIFYING,
    BLENDED
};

class Species;

class Population
{
    /////////////////////
    // Members
    /////////////////////

private:

    // The innovation database
    InnovationDatabase m_InnovationDatabase;

    // next genome ID
    unsigned int m_NextGenomeID;

    // next species ID
    unsigned int m_NextSpeciesID;

    ////////////////////////////
    // Phased searching members

    // The current mode of search
    SearchMode m_SearchMode;

    // The current Mean Population Complexity
    double m_CurrentMPC;

    // The MPC from the previous generation (for comparison)
    double m_OldMPC;

    // The base MPC (for switching between complexifying/simplifying phase)
    double m_BaseMPC;

    // Adjusts each species's fitness
    void AdjustFitness();

    // Calculates how many offspring each genome should have
    void CountOffspring();

    // Empties all species
    void ResetSpecies();

    // Updates the species
    void UpdateSpecies();

    // Calculates the current mean population complexity
    void CalculateMPC();


    // best fitness ever achieved
    double m_BestFitnessEver;

    // Keep a local copy of the best ever genome found in the run
    Genome m_BestGenome;
    Genome m_BestGenomeEver;

    // Number of generations since the best fitness changed
    unsigned int m_GensSinceBestFitnessLastChanged;

    // Number of evaluations since the best fitness changed
    unsigned int m_EvalsSinceBestFitnessLastChanged;

    // How many generations passed until the last change of MPC
    unsigned int m_GensSinceMPCLastChanged;

    // The initial list of genomes
    std::vector<Genome> m_Genomes;

public:

    // The archive
    std::vector<Genome> m_GenomeArchive;

    // Random number generator
    RNG m_RNG;

    // Evolution parameters
    Parameters m_Parameters;

    // Current generation
    unsigned int m_Generation;

    // The list of species
    std::vector<Species> m_Species;


    ////////////////////////////
    // Constructors
    ////////////////////////////

    // Initializes a population from a seed genome G. Then it initializes all weights
    // To small numbers between -R and R.
    // The population size is determined by GlobalParameters.PopulationSize
    Population(const Genome& a_G, const Parameters& a_Parameters,
    		   bool a_RandomizeWeights, double a_RandomRange, int a_RNG_seed);


    // Loads a population from a file.
    Population(const char* a_FileName);

    ////////////////////////////
    // Destructor
    ////////////////////////////

    // TODO: move all header code into the source file,
    // make as much private members as possible

    ////////////////////////////
    // Methods
    ////////////////////////////
    int ResizeInputs(int a_Size);
    void DisconnectInputs(const std::vector<int> &input_idxs);
    void RandomizeOutgoingWeights(const std::vector<int> &input_idxs);
    #ifdef USE_BOOST_PYTHON
    void DisconnectInputs_py(const py::list &input_idxs);
    void RandomizeOutgoingWeights_py(const py::list &input_idxs);
    #endif

    // Separates the population into species based on compatibility distance
    void Speciate();

    ////////////////////////////
    // Island methods

    // Get N=quantity of this population's best genomes
    // The genomes are selected by iteratively picking the best genome of each specie
    vector<Genome> GetBestGenomesBySpecies(int quantity);

    // Replace random genomes from this population, chosen randomly, with those received as input
    void ReplaceGenomes(std::vector<Genome> replacements);

    // Access
    SearchMode GetSearchMode() const { return m_SearchMode; }
    double GetCurrentMPC() const { return m_CurrentMPC; }
    double GetBaseMPC() const { return m_BaseMPC; }

    unsigned int NumGenomes() const
    {
    	unsigned int num=0;
    	for(unsigned int i=0; i<m_Species.size(); i++)
    	{
    		num += m_Species[i].m_Individuals.size();
    	}
    	return num;
    }

    unsigned int GetGeneration() const { return m_Generation; }
    double GetBestFitnessEver() const { return m_BestFitnessEver; }
    Genome GetBestGenome() const
    {
        double best = std::numeric_limits<double>::min();
        int idx_species = 0;
        int idx_genome = 0;
        for(unsigned int i=0; i<m_Species.size(); i++)
        {
            for(unsigned int j=0; j<m_Species[i].m_Individuals.size(); j++)
            {
                if (m_Species[i].m_Individuals[j].GetFitness() > best)
                {
                    best = m_Species[i].m_Individuals[j].GetFitness();
                    idx_species = i;
                    idx_genome = j;
                }
            }
        }

        return m_Species[idx_species].m_Individuals[idx_genome];
    }

    unsigned int GetStagnation() const { return m_GensSinceBestFitnessLastChanged; }
    unsigned int GetMPCStagnation() const { return m_GensSinceMPCLastChanged; }

    unsigned int GetNextGenomeID() const { return m_NextGenomeID; }
    unsigned int GetNextSpeciesID() const { return m_NextSpeciesID; }
    void IncrementNextGenomeID() { m_NextGenomeID++; }
    void IncrementNextSpeciesID() { m_NextSpeciesID++; }

    Genome& AccessGenomeByIndex(unsigned int const a_idx);
    Genome& AccessGenomeByID(unsigned int const a_id);

    InnovationDatabase& AccessInnovationDatabase() { return m_InnovationDatabase; }

    // Sorts each species's genomes by fitness
    void Sort();

    // Performs one generation and reproduces the genomes
    void Epoch();

    // Saves the whole population to a file
    void Save(const char* a_FileName);

    //////////////////////
    // NEW STUFF
    std::vector<Species> m_TempSpecies; // useful in reproduction


    //////////////////////
    // Real-Time methods

    // Estimates the estimated average fitness for all species
    //void EstimateAllAverages();

    // Reproduce the population champ only
    //Genome ReproduceChamp();

    // Choose the parent species that will reproduce
    // This is a real-time version of fitness sharing
    // Returns the species index
    unsigned int ChooseParentSpecies();

    // Removes worst member of the whole population that has been around for a minimum amount of time
    // returns the genome that was just deleted (may be useful)
    Genome RemoveWorstIndividual();

    // The main reaitime tick. Analog to Epoch(). Replaces the worst evaluated individual with a new one.
    // Returns a pointer to the new baby.
    // and copies the genome that was deleted to a_geleted_genome
    Genome* Tick(Genome& a_deleted_genome);

    // Takes an individual and puts it in its apropriate species
    // Useful in realtime when the compatibility treshold changes
    void ReassignSpecies(unsigned int a_genome_idx);

    unsigned int m_NumEvaluations;



    ///////////////////////////////
    // Novelty search

    // A pointer to the archive of PhenotypeBehaviors
    // Necessary to contain derived custom classes.
    std::vector< PhenotypeBehavior >* m_BehaviorArchive;

    // Call this function to allocate memory for your custom
    // behaviors. This initializes everything.
    void InitPhenotypeBehaviorData(std::vector< PhenotypeBehavior >* a_population,
                                   std::vector< PhenotypeBehavior >* a_archive);

    // This is the main method performing novelty search.
    // Performs one reproduction and assigns novelty scores
    // based on the current population and the archive.
    // If a successful behavior was encountered, returns true
    // and the genome a_SuccessfulGenome is overwritten with the
    // genome generating the successful behavior
    bool NoveltySearchTick(Genome& a_SuccessfulGenome);

    double ComputeSparseness(Genome& genome);

    // counters for archive stagnation
    unsigned int m_GensSinceLastArchiving;
    unsigned int m_QuickAddCounter;
};

} // namespace NEAT

#endif
