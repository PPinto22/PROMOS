#include "MultiNEAT/Genome.h"
#include "MultiNEAT/Population.h"
#include "MultiNEAT/NeuralNetwork.h"
#include "MultiNEAT/Parameters.h"
#include "MultiNEAT/Substrate.h"
#include "MultiNEAT/Random.h"
#include "csv/csv.h"

#include <iostream>
#include <boost/asio.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

using namespace NEAT;


const string DATA_FILE_PATH = "../data/data_micro.csv";
const string OUT_DIR = "../results";
const int GENERATIONS = 50;

std::vector< std::map< std::string, double > > readData(const string file_path){
    io::CSVReader<11, io::trim_chars<' ','\t','\"'>, io::no_quote_escape<';'>, io::throw_on_overflow, io::no_comment> in(DATA_FILE_PATH);
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
    for(auto row: data){
        for(auto col: row) {
            cout << col.first << ":" << col.second << endl;
        }
    }
    return data;
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
    params.EliteFraction = 1;
    params.CrossoverRate = 0.5;
    params.MutateNeuronTraitsProb = 0;
    params.MutateLinkTraitsProb = 0;

    Genome g(0, 10, 0, 1, false, ActivationFunction::UNSIGNED_SIGMOID, ActivationFunction::UNSIGNED_SIGMOID, 0, params, 5);
    Population pop(g, params, true, 1.0, 0);

    readData(DATA_FILE_PATH);

    return 0;
}
