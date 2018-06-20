#ifndef PHENOTYPE_H
#define PHENOTYPE_H

#ifdef USE_BOOST_PYTHON

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/tuple.hpp>
#include <math.h>
#include <cmath>

namespace py = boost::python;

#endif

#include <vector>
#include "Genes.h"

namespace NEAT {

    class Connection {
    public:
        int m_source_neuron_idx;       // index of source neuron
        int m_target_neuron_idx;       // index of target neuron
        double m_weight;                               // weight of the connection
        double m_signal;                               // weight * input signal

        bool m_recur_flag; // recurrence flag for displaying purposes
        // can be ignored

        // Hebbian learning parameters
        // Ignored in case there is no lifetime learning
        double m_hebb_rate;
        double m_hebb_pre_rate;

        // comparison operator (nessesary for boost::python)
        bool operator==(Connection const &other) const {
            if ((m_source_neuron_idx == other.m_source_neuron_idx) &&
                (m_target_neuron_idx == other.m_target_neuron_idx)) /*&&
            (m_weight == other.m_weight) &&
            (m_recur_flag == other.m_recur_flag))*/
                return true;
            else
                return false;
        }

    };

    class Neuron {
    public:
        double m_activesum{0};  // the synaptic input
        double m_activation{0}; // the synaptic input passed through the activation function

        double m_a{1}, m_b{0}, m_timeconst{0}, m_bias{0}; // misc parameters
        double m_membrane_potential{0}; // used in leaky integrator mode
        ActivationFunction m_activation_function_type;

        // displaying and stuff
        double m_x, m_y, m_z;
        double m_sx, m_sy, m_sz;
        std::vector<double> m_substrate_coords;
        double m_split_y;
        NeuronType m_type;

        // the sensitivity matrix of this neuron (for RTRL learning)
        std::vector<std::vector<double> > m_sensitivity_matrix;

        // comparison operator (nessesary for boost::python)
        bool operator==(Neuron const &other) const {
            if ((m_type == other.m_type) &&
                (m_split_y == other.m_split_y) &&
                (m_activation_function_type == other.m_activation_function_type)// &&
                //(this == other.this))
                    )
                return true;
            else
                return false;
        }

        void ApplyActivationFunction();
    };

    class NeuralNetwork {
        /////////////////////
        // RTRL variables
        double m_total_error;

        // Always the size of m_connections
        std::vector<double> m_total_weight_change;
        /////////////////////

        // returns the index if that connection exists or -1 otherwise
        int ConnectionExists(int a_to, int a_from);

    public:

        unsigned int m_num_inputs, m_num_outputs;
        std::vector<Connection> m_connections; // array size - number of connections
        std::vector<Neuron> m_neurons;

        // List< Pair<Neuron_idx, List<Connection>> >
        std::vector<std::pair<int, std::vector<Connection *>>> m_ordered_connections;

        NeuralNetwork(bool a_Minimal); // if given false, the constructor will create a standard XOR network topology.
        NeuralNetwork();

        void InitRTRLMatrix(); // initializes the sensitivity cube for RTRL learning.
        // assumes that neuron and connection data are already initialized

        void ActivateFast();          // assumes unsigned sigmoids everywhere.
        void Activate();              // any activation functions are supported
        void ActivateUseInternalBias(); // like Activate() but uses m_bias as well
        void ActivateLeaky(double step); // activates in leaky integrator mode
        void FeedForward(); // Activates the whole network with a single pass (Requires SortConnections)

        // Sort connections from input to output (sets m_ordered_connections)
        // only works for Feed Forward networks, with no cycles
        void SortConnections();


        void RTRL_update_gradients();

        void RTRL_update_error(double a_target);

        void RTRL_update_weights();   // performs the backprop step

        // Hebbian learning
        void Adapt(Parameters &a_Parameters);

        void Flush();     // clears all activations
        void FlushCube(); // clears the sensitivity cube

        void Input(std::vector<double> &a_Inputs);

        int GetConnectionsQty();

        int GetNeuronsQty();

#ifdef USE_BOOST_PYTHON

        void Input_python_list(const py::list &a_Inputs);

        void Input_numpy(const py::numeric::array &a_Inputs);

#endif

        std::vector<double> Output();

        // accessor methods
        void AddNeuron(const Neuron &a_n) { m_neurons.push_back(a_n); }

        void AddConnection(const Connection &a_c) { m_connections.push_back(a_c); }

        Connection GetConnectionByIndex(unsigned int a_idx) const {
            return m_connections[a_idx];
        }

        Neuron GetNeuronByIndex(unsigned int a_idx) const {
            return m_neurons[a_idx];
        }

        void SetInputOutputDimentions(const unsigned int a_i, const unsigned int a_o) {
            m_num_inputs = a_i;
            m_num_outputs = a_o;
        }

        unsigned int NumInputs() const {
            return m_num_inputs;
        }

        unsigned int NumOutputs() const {
            return m_num_outputs;
        }

        // clears the network and makes it a minimal one
        void Clear() {
            m_neurons.clear();
            m_connections.clear();
            m_total_weight_change.clear();
            SetInputOutputDimentions(0, 0);
        }

        double GetConnectionLenght(Neuron source, Neuron target) {
            double dist = 0.0;
            for (unsigned int i = 0; i < source.m_substrate_coords.size(); i++) {
                dist += (target.m_substrate_coords[i] - source.m_substrate_coords[i]) *
                        (target.m_substrate_coords[i] - source.m_substrate_coords[i]);
            }
            return dist;
        }

        double GetTotalConnectionLength() {
            return m_connections.size();
        }

        // one-shot save/load
        void Save(const char *a_filename);

        bool Load(const char *a_filename);

        // save/load from already opened files for reading/writing
        void Save(FILE *a_file);

        bool Load(std::ifstream &a_DataFile);
    };

}; // namespace NEAT




#endif
