#include "MultiNEAT/Genome.h"
#include "MultiNEAT/Population.h"
#include "MultiNEAT/NeuralNetwork.h"
#include "MultiNEAT/Parameters.h"
#include "MultiNEAT/Substrate.h"
#include "MultiNEAT/Random.h"
#include "csv/csv.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <functional>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace NEAT;


const char* DATA_FILE_PATH = "../data/data_micro.csv";
const char* OUT_DIR = "../results/TEMP";
const int GENERATIONS = 5;


std::vector< std::map< std::string, double > > readData(const string& filePath){
    io::CSVReader<11, io::trim_chars<' ','\t','\"'>, io::no_quote_escape<';'>, io::throw_on_overflow, io::no_comment> in(filePath);
    in.read_header(io::ignore_extra_column, "target", "regioncontinent", "idcampaign", "idpartner", "idverticaltype",
                   "idbrowser", "idaffmanager", "idapplication", "idoperator", "accmanager", "country_name");

    double target, regioncontinent, idcampaign, idpartner, idverticaltype, idbrowser;
    double idaffmanager, idapplication, idoperator, accmanager, country_name;

    std::vector< std::map<std::string, double> > data;

    while(in.read_row(target, regioncontinent, idcampaign, idpartner, idverticaltype, idbrowser,
                      idaffmanager, idapplication, idoperator, accmanager, country_name)){
        std::map<std::string, double> row;
        row["target"] = target;
        row["regioncontinent"] = regioncontinent;
        row["idcampaign"] = idcampaign;
        row["idpartner"] = idpartner;
        row["idverticaltype"] = idverticaltype;
        row["idbrowser"] = idbrowser;
        row["idaffmanager"] = idaffmanager;
        row["idapplication"] = idapplication;
        row["idoperator"] = idoperator;
        row["accmanager"] = accmanager;
        row["country_name"] = country_name;
        data.push_back(row);
    }

    return data;
}

double evaluateGenome(Genome& genome, std::vector< std::map<std::string, double> >& data){
    NeuralNetwork net;
    genome.BuildPhenotype(net);

    std::vector<double> predictions(data.size());

    for (auto it = data.begin(); it != data.end(); it++) {
        long i = std::distance(data.begin(), it);
        std::map<std::string, double>& row = *it;
        std::vector<double> input(10);
        input.at(0) = row["regioncontinent"];
        input.at(1) = row["idcampaign"];
        input.at(2) = row["idpartner"];
        input.at(3) = row["idverticaltype"];
        input.at(4) = row["idbrowser"];
        input.at(5) = row["idaffmanager"];
        input.at(6) = row["idapplication"];
        input.at(7) = row["idoperator"];
        input.at(8) = row["accmanager"];
        input.at(9) = row["country_name"];
        net.Flush();
        net.Input(input);
        net.Activate();
        auto output = net.Output();
        predictions.at(static_cast<unsigned long>(i)) = output[0];
        net.Flush();
    }

    double error = 0;
    for (int i=0; i<data.size(); i++){
        error += fabs(data[i]["target"] - predictions[i]);
    }
    double fitness = 1.0 / error;

    genome.SetFitness(fitness);
    genome.SetEvaluated();

    return fitness;
}

std::vector<std::pair<Genome*, double>> evaluatePopulation(Population& pop, std::vector<std::map<std::string, double>>& data){
    std::vector<std::pair<Genome*, double>> evaluations;
    for(Species& s: pop.m_Species){
        for(Genome& g: s.m_Individuals){
            double fitness = evaluateGenome(g, data);
            evaluations.push_back(std::make_pair(&g,fitness));
        }
    }
    return evaluations;
}

std::string msToTimeString(long ms){
    long total_seconds = ms / 1000;
    long total_minutes = total_seconds / 60;
    long total_hours = total_minutes / 60;

    std::ostringstream oss;
    oss <<        std::setfill('0') << std::setw(2) << total_hours
        << ":" << std::setfill('0') << std::setw(2) << total_minutes % 60
        << ":" << std::setfill('0') << std::setw(2) << total_seconds % 60
        << "." << std::setfill('0') << std::setw(3) << ms % 1000;
    return oss.str();
}

std::string getCurrentDateString(){
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y_%H:%M:%S");
    return oss.str();
}

void writeResults(const char* filePath, int generations, Genome& best_genome, NeuralNetwork& best_net,
                  double best_fitness, long ms_total, long ms_eval, long ms_ea) {
    std::ofstream results_file;
    results_file.open(filePath);
    results_file << "Generations: " << generations << endl;
    results_file << "Best fitness: " << best_fitness << endl;
    results_file << "Network neurons: " << best_net.m_neurons.size() << endl;
    results_file << "Network connections: " << best_net.m_connections.size() << endl;
    results_file << "Execution time: " << msToTimeString(ms_total) << endl;
    results_file << "Total evaluation time: " << msToTimeString(ms_eval) << endl;
    results_file << "Total EA time: " << msToTimeString(ms_ea) << endl;
    results_file.close();
}


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
    params.EliteFraction = 0.1;
    params.CrossoverRate = 0.5;
    params.MutateNeuronTraitsProb = 0;
    params.MutateLinkTraitsProb = 0;

    long ms_eval = 0;
    long ms_ea = 0;
    auto start = chrono::system_clock::now();

    Genome g(0, 10, 0, 1, false, ActivationFunction::UNSIGNED_SIGMOID, ActivationFunction::UNSIGNED_SIGMOID, 0, params, 5);
    Population pop(g, params, true, 1.0, 0);

    auto data = readData(DATA_FILE_PATH);
    std::pair<Genome, double> all_time_best; // pair<Genome, fitness>

    for(int gen=0; gen<GENERATIONS; gen++){

        auto start_eval = chrono::system_clock::now();
        std::vector<std::pair<Genome*, double>> evaluations = evaluatePopulation(pop, data);
        auto end_eval = chrono::system_clock::now();
        ms_eval += chrono::duration_cast<chrono::milliseconds>(end_eval - start_eval).count();

        std::pair<Genome*, double> best = *std::max_element(
                evaluations.begin(), evaluations.end(),
                [](std::pair<Genome*, double>& pair1, std::pair<Genome*, double>& pair2) -> bool {
                    return pair1.second < pair2.second;
                });

        cout << printf("Best genome of generation %d: %lf", gen, best.second) << endl;

        all_time_best = best.second > all_time_best.second ? std::make_pair(Genome(*best.first), best.second) : all_time_best;

        auto start_ea = chrono::system_clock::now();
        pop.Epoch();
        auto end_ea = chrono::system_clock::now();
        ms_ea += chrono::duration_cast<chrono::milliseconds>(end_ea - start_ea).count();
    }

    auto end = chrono::system_clock::now();
    long ms_total = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    char fileName[60];
    sprintf(fileName, "%s/neat_cpp_%s.txt", OUT_DIR, getCurrentDateString().c_str());
    NeuralNetwork net;
    all_time_best.first.BuildPhenotype(net);

    writeResults(fileName, GENERATIONS, all_time_best.first, net, all_time_best.second, ms_total, ms_eval, ms_ea);

    return 0;
}
