#include <math.h>
#include <float.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include "NeuralNetwork.h"
#include "Assert.h"
#include "Utils.hpp"

//#define NULL 0
#define sqr(x) ((x)*(x))
#define LEARNING_RATE 0.0001

namespace NEAT {

/////////////////////////////////////
// The set of activation functions //
/////////////////////////////////////


    inline double af_sigmoid_unsigned(double aX, double aSlope, double aShift) {
        return 1.0 / (1.0 + exp(-aSlope * aX - aShift));
    }

    inline double af_sigmoid_signed(double aX, double aSlope, double aShift) {
        double tY = af_sigmoid_unsigned(aX, aSlope, aShift);
        return (tY - 0.5) * 2.0;
    }

    inline double af_tanh(double aX, double aSlope, double aShift) {
        return tanh(aX * aSlope);
    }

    inline double af_tanh_cubic(double aX, double aSlope, double aShift) {
        return tanh(aX * aX * aX * aSlope);
    }

    inline double af_step_signed(double aX, double aShift) {
        double tY;
        if (aX > aShift) {
            tY = 1.0;
        } else {
            tY = -1.0;
        }

        return tY;
    }

    inline double af_step_unsigned(double aX, double aShift) {
        if (aX > (0.5 + aShift)) {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    inline double af_gauss_signed(double aX, double aSlope, double aShift) {
        double tY = exp(-aSlope * aX * aX + aShift); // TODO: Need separate a, b per activation function
        return (tY - 0.5) * 2.0;
    }

    inline double af_gauss_unsigned(double aX, double aSlope, double aShift) {
        return exp(-aSlope * aX * aX + aShift);
    }

    inline double af_abs(double aX, double aShift) {
        return ((aX + aShift) < 0.0) ? -(aX + aShift) : (aX + aShift);
    }

    inline double af_sine_signed(double aX, double aFreq, double aShift) {
        return sin(aX * aFreq + aShift);
    }

    inline double af_sine_unsigned(double aX, double aFreq, double aShift) {
        double tY = sin((aX * aFreq + aShift));
        return (tY + 1.0) / 2.0;
    }


    inline double af_linear(double aX, double aShift) {
        return (aX + aShift);
    }


    inline double af_relu(double aX) {
        return (aX > 0) ? aX : 0;
    }


    inline double af_softplus(double aX) {
        return log(1 + exp(aX));
    }


    double unsigned_sigmoid_derivative(double x) {
        return x * (1 - x);
    }

    double tanh_derivative(double x) {
        return 1 - x * x;
    }


///////////////////////////////////////
// Neuron class implementation
///////////////////////////////////////
    void Neuron::ApplyActivationFunction() {
        switch (m_activation_function_type) {
            case SIGNED_SIGMOID:
                m_activation = af_sigmoid_signed(m_activesum, m_a, m_b);
                break;
            case UNSIGNED_SIGMOID:
                m_activation = af_sigmoid_unsigned(m_activesum, m_a, m_b);
                break;
            case TANH:
                m_activation = af_tanh(m_activesum, m_a, m_b);
                break;
            case TANH_CUBIC:
                m_activation = af_tanh_cubic(m_activesum, m_a, m_b);
                break;
            case SIGNED_STEP:
                m_activation = af_step_signed(m_activesum, m_b);
                break;
            case UNSIGNED_STEP:
                m_activation = af_step_unsigned(m_activesum, m_b);
                break;
            case SIGNED_GAUSS:
                m_activation = af_gauss_signed(m_activesum, m_a, m_b);
                break;
            case UNSIGNED_GAUSS:
                m_activation = af_gauss_unsigned(m_activesum, m_a, m_b);
                break;
            case ABS:
                m_activation = af_abs(m_activesum, m_b);
                break;
            case SIGNED_SINE:
                m_activation = af_sine_signed(m_activesum, m_a, m_b);
                break;
            case UNSIGNED_SINE:
                m_activation = af_sine_unsigned(m_activesum, m_a, m_b);
                break;
            case LINEAR:
                m_activation = af_linear(m_activesum, m_b);
                break;
            case RELU:
                m_activation = af_relu(m_activesum);
                break;
            case SOFTPLUS:
                m_activation = af_softplus(m_activesum);
                break;
            default:
                m_activation = af_sigmoid_unsigned(m_activesum, m_a, m_b);
                break;
        }
    }

///////////////////////////////////////
// Neural network class implementation
///////////////////////////////////////
    NeuralNetwork::NeuralNetwork(bool a_Minimal) {
        if (!a_Minimal) {
            // build an XOR network

            // The input neurons are 3 // indexes 0 1 2
            Neuron t_i1, t_i2, t_i3;

            // The output neuron       // index 3
            Neuron t_o1;

            // The hidden neuron       // index 4
            Neuron t_h1;

            m_neurons.push_back(t_i1);
            m_neurons.push_back(t_i2);
            m_neurons.push_back(t_i3);
            m_neurons.push_back(t_o1);
            m_neurons.push_back(t_h1);

            // The connections
            Connection t_c;

            t_c.m_source_neuron_idx = 0;
            t_c.m_target_neuron_idx = 3;
            t_c.m_weight = 0;
            m_connections.push_back(t_c);

            t_c.m_source_neuron_idx = 1;
            t_c.m_target_neuron_idx = 3;
            t_c.m_weight = 0;
            m_connections.push_back(t_c);

            t_c.m_source_neuron_idx = 2;
            t_c.m_target_neuron_idx = 3;
            t_c.m_weight = 0;
            m_connections.push_back(t_c);

            t_c.m_source_neuron_idx = 0;
            t_c.m_target_neuron_idx = 4;
            t_c.m_weight = 0;
            m_connections.push_back(t_c);

            t_c.m_source_neuron_idx = 1;
            t_c.m_target_neuron_idx = 4;
            t_c.m_weight = 0;
            m_connections.push_back(t_c);

            t_c.m_source_neuron_idx = 2;
            t_c.m_target_neuron_idx = 4;
            t_c.m_weight = 0;
            m_connections.push_back(t_c);

            t_c.m_source_neuron_idx = 4;
            t_c.m_target_neuron_idx = 3;
            t_c.m_weight = 0;
            m_connections.push_back(t_c);

            m_num_inputs = 3;
            m_num_outputs = 1;

            // Initialize the network's weights (make them random)
            for (unsigned int i = 0; i < m_connections.size(); i++) {
                m_connections[i].m_weight = ((double) rand() / (double) RAND_MAX)
                                            - 0.5;
            }

            // clean up other neuron data as well
            for (unsigned int i = 0; i < m_neurons.size(); i++) {
                m_neurons[i].m_a = 1;
                m_neurons[i].m_b = 0;
                m_neurons[i].m_timeconst = m_neurons[i].m_bias =
                m_neurons[i].m_membrane_potential = 0;
            }

            InitRTRLMatrix();
        } else {
            // an empty network
            m_num_inputs = m_num_outputs = 0;
            m_total_error = 0;
            // clean up other neuron data as well
            for (unsigned int i = 0; i < m_neurons.size(); i++) {
                m_neurons[i].m_a = 1;
                m_neurons[i].m_b = 0;
                m_neurons[i].m_timeconst = m_neurons[i].m_bias =
                m_neurons[i].m_membrane_potential = 0;
            }
            Clear();
        }
    }

    NeuralNetwork::NeuralNetwork() {
        // an empty network
        m_num_inputs = m_num_outputs = 0;
        m_total_error = 0;
        // clean up other neuron data as well
        for (unsigned int i = 0; i < m_neurons.size(); i++) {
            m_neurons[i].m_a = 1;
            m_neurons[i].m_b = 0;
            m_neurons[i].m_timeconst = m_neurons[i].m_bias =
            m_neurons[i].m_membrane_potential = 0;
        }
        Clear();
    }

    void NeuralNetwork::InitRTRLMatrix() {
        // Allocate memory for the neurons sensitivity matrices.
        for (unsigned int i = 0; i < m_neurons.size(); i++) {
            m_neurons[i].m_sensitivity_matrix.resize(m_neurons.size()); // first dimention
            for (unsigned int j = 0; j < m_neurons.size(); j++) {
                m_neurons[i].m_sensitivity_matrix[j].resize(m_neurons.size()); // second dimention
            }
        }

        // now clear it
        FlushCube();
        // clear out the other RTRL stuff as well
        m_total_error = 0;
        m_total_weight_change.resize(m_connections.size());
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_total_weight_change[i] = 0;
        }
    }

    void NeuralNetwork::ActivateFast() {
        // Loop connections. Calculate each connection's output signal.
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_connections[i].m_signal =
                    m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                    * m_connections[i].m_weight;
        }
        // Loop the connections again. This time add the signals to the target neurons.
        // This will largely require out of order memory writes. This is the one loop where
        // this will happen.
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                    m_connections[i].m_signal;
        }
        // Now loop nodes_activesums, pass the signals through the activation function
        // and store the result back to nodes_activations
        // also skip inputs since they do not get an activation
        for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++) {
            double x = m_neurons[i].m_activesum;
            m_neurons[i].m_activesum = 0;
            // Apply the activation function
            double y = 0.0;
            y = af_sigmoid_unsigned(x, m_neurons[i].m_a, m_neurons[i].m_b);
            m_neurons[i].m_activation = y;
        }
    }

    void NeuralNetwork::Activate() {
        // Loop connections. Calculate each connection's output signal.
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_connections[i].m_signal =
                    m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                    * m_connections[i].m_weight;
        }
        // Loop the connections again. This time add the signals to the target neurons.
        // This will largely require out of order memory writes. This is the one loop where
        // this will happen.
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                    m_connections[i].m_signal;
        }
        // Now loop nodes_activesums, pass the signals through the activation function
        // and store the result back to nodes_activations
        // also skip inputs since they do not get an activation
        for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++) {
            m_neurons[i].ApplyActivationFunction();
        }

    }

    void NeuralNetwork::ActivateUseInternalBias() {
        // Loop connections. Calculate each connection's output signal.
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_connections[i].m_signal =
                    m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                    * m_connections[i].m_weight;
        }
        // Loop the connections again. This time add the signals to the target neurons.
        // This will largely require out of order memory writes. This is the one loop where
        // this will happen.
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                    m_connections[i].m_signal;
        }
        // Now loop nodes_activesums, pass the signals through the activation function
        // and store the result back to nodes_activations
        // also skip inputs since they do not get an activation
        for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++) {
            m_neurons[i].m_activesum += m_neurons[i].m_bias;
            m_neurons[i].ApplyActivationFunction();
        }

    }

    void NeuralNetwork::ActivateLeaky(double a_dtime) {
        // Loop connections. Calculate each connection's output signal.
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_connections[i].m_signal =
                    m_neurons[m_connections[i].m_source_neuron_idx].m_activation
                    * m_connections[i].m_weight;
        }
        // Loop the connections again. This time add the signals to the target neurons.
        // This will largely require out of order memory writes. This is the one loop where
        // this will happen.
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
                    m_connections[i].m_signal;
        }
        // Now we have the leaky integrator step for the neurons
        for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++) {
            double t_const = a_dtime / m_neurons[i].m_timeconst;
            m_neurons[i].m_membrane_potential = (1.0 - t_const)
                                                * m_neurons[i].m_membrane_potential
                                                + t_const * m_neurons[i].m_activesum;
        }
        // Now loop nodes_activesums, pass the signals through the activation function
        // and store the result back to nodes_activations
        // also skip inputs since they do not get an activation
        for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++) {
            m_neurons[i].m_activesum = m_neurons[i].m_membrane_potential + m_neurons[i].m_bias;
            m_neurons[i].ApplyActivationFunction();
        }

    }

    void NeuralNetwork::SortConnections() {
        m_ordered_connections.clear();

        // Map<Index, Connection*> of connections not yet transversed
        // Initially, it's every connection
        std::unordered_map<int, Connection *> t_connections;
        // Set of neurons with dependencies (i.e., with at least 1 incoming connection)
        std::unordered_set<int> t_neurons_w_dependencies;

        // Iterate through m_connections and copy pointers of each connection to t_connections
        // Also, add neurons that appear as targets in any connection to t_neurons_w_dependencies
        for (auto it = m_connections.begin(); it != m_connections.end(); it++) {
            long i = std::distance(m_connections.begin(), it);
            Connection& c = *it;
            t_connections.insert(std::make_pair<int, Connection *>(i, &c));
            t_neurons_w_dependencies.insert(c.m_target_neuron_idx);
        }

        // For every neuron index KEY, save the list of connections where the target is KEY
        // (except inputs, which are never a target)
        std::unordered_map<int, std::vector<Connection *>> t_target_connections;
        for (long i = m_num_inputs; i < m_neurons.size(); i++) {
            t_target_connections.insert(
                    std::make_pair(i, std::vector<Connection *>())
            );
        }

        // While there are connections not yet transversed
        while (!t_connections.empty()) {
            long t_initial_size = t_connections.size();
            // Iterate through all possible connections
            for (auto it = t_connections.begin(); it != t_connections.end(); /* no increment */) {
                Connection &c = *((*it).second);
                Neuron &source = m_neurons[c.m_source_neuron_idx];
                Neuron &target = m_neurons[c.m_target_neuron_idx];

                // If source has no dependencies
                if (t_neurons_w_dependencies.find(c.m_source_neuron_idx) == t_neurons_w_dependencies.end()) {
                    // Add connection to the list of connections whose target is this connection's target
                    t_target_connections[c.m_target_neuron_idx].push_back(&c);

                    // Remove connection from t_connections
                    it = t_connections.erase(it);

                    // Check if target has any more dependencies
                    // That is, check that it does not appear as a target in any connection in t_connections
                    bool t_has_dependencies = find_if(
                            t_connections.begin(), t_connections.end(),
                            [&c](const std::pair<int, Connection *> &lambda_c) {
                                return (lambda_c.second)->m_target_neuron_idx == c.m_target_neuron_idx;
                            }) != t_connections.end();
                    // If target has no more dependencies, append it and
                    // the list of connections that target it to m_ordered_connections.
                    // Also, add it to t_neurons_w_dependencies
                    if (!t_has_dependencies) {
                        m_ordered_connections.emplace_back(c.m_target_neuron_idx, t_target_connections[c.m_target_neuron_idx]);
                        t_neurons_w_dependencies.erase(c.m_target_neuron_idx);
                    }
                } else { // Source has not yet received all inputs; skip
                    ++it;
                }
            }

            long t_final_size = t_connections.size();
            // If no connection was removed, then we're stuck in an infinite loop.
            // Caused by a loop in the network -- this method only works for feed forward networks!
            if (t_final_size == t_initial_size) {
                throw std::domain_error("Trying to sort the connections of a network that has loops");
            }
        }
    }

    void NeuralNetwork::FeedForward() {
        if (m_ordered_connections.empty()) {
            SortConnections();
        }
        for(long i = 0; i < m_ordered_connections.size(); i++){
            Neuron& target = m_neurons[m_ordered_connections[i].first];
            std::vector<Connection *>& connections = m_ordered_connections[i].second;
            for(Connection* c : connections){
                // Propagate signal from source to target
                Neuron& source = m_neurons[c->m_source_neuron_idx];
                c->m_signal = source.m_activation * c->m_weight;
                target.m_activesum += c->m_signal;
            }
            // Apply bias and activation function
            target.m_activesum += target.m_bias;
            target.ApplyActivationFunction();
        }
    }

    void NeuralNetwork::Flush() {
        for (unsigned int i = 0; i < m_neurons.size(); i++) {
            m_neurons[i].m_activation = 0;
            m_neurons[i].m_activesum = 0;
            m_neurons[i].m_membrane_potential = 0;
        }
    }

    void NeuralNetwork::FlushCube() {
        // clear the cube
        for (unsigned int i = 0; i < m_neurons.size(); i++)
            for (unsigned int j = 0; j < m_neurons.size(); j++)
                for (unsigned int k = 0; k < m_neurons.size(); k++)
                    m_neurons[k].m_sensitivity_matrix[i][j] = 0;
    }

    void NeuralNetwork::Input(const std::vector<double> &a_Inputs) {
        unsigned mx = a_Inputs.size();
        if (mx != m_num_inputs) {
            throw std::invalid_argument( std::string("Number of inputs received (") + to_string(mx) + ") " +
                                         "does not match the network's number of inputs (" + to_string(m_num_inputs) + ")." );
        }

        for (unsigned int i = 0; i < mx; i++) {
            m_neurons[i].m_activation = a_Inputs[i];
        }
    }

    int NeuralNetwork::NumConnections(){
        return m_connections.size();
    }

    int NeuralNetwork::NumNeurons(){
        return m_neurons.size();
    }

    int NeuralNetwork::NumHiddenNeurons(){
        return m_neurons.size() - m_num_inputs - m_num_outputs;
    }

#ifdef USE_BOOST_PYTHON

    void NeuralNetwork::Input_python_list(const py::list &a_Inputs) {
        unsigned len = py::len(a_Inputs);
        if (len != m_num_inputs) {
            throw std::invalid_argument( std::string("Number of inputs received (") + to_string(len) + ") " +
                                         "does not match the network's number of inputs (" + to_string(m_num_inputs) + ")." );
        }
        for (int i = 0; i < len; i++) {
            m_neurons[i].m_activation = py::extract<double>(a_Inputs[i]);
        }
    }

    void NeuralNetwork::Input_numpy(const py::numeric::array &a_Inputs) {
      unsigned len = py::len(a_Inputs);
      if (len != m_num_inputs) {
          throw std::invalid_argument( std::string("Number of inputs received (") + to_string(len) + ") " +
                                       "does not match the network's number of inputs (" + to_string(m_num_inputs) + ")." );
      }
      for (int i = 0; i < len; i++) {
          m_neurons[i].m_activation = py::extract<double>(a_Inputs[i]);
      }
    }

#endif

    std::vector<double> NeuralNetwork::Output() {
        std::vector<double> t_output;

        for (int i = 0; i < m_num_outputs; i++) {
            t_output.push_back(m_neurons[i + m_num_inputs].m_activation);
        }
        return t_output;
    }

    void NeuralNetwork::Adapt(Parameters &a_Parameters) {
        // find max absolute magnitude of the weight
        double t_max_weight = -999999999;
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            if (fabs(m_connections[i].m_weight) > t_max_weight) {
                t_max_weight = fabs(m_connections[i].m_weight);
            }
        }

        for (unsigned int i = 0; i < m_connections.size(); i++) {
            /////////////////////////////////////
            // modify weight of that connection
            ////
            double t_incoming_neuron_activation =
                    m_neurons[m_connections[i].m_source_neuron_idx].m_activation;
            double t_outgoing_neuron_activation =
                    m_neurons[m_connections[i].m_target_neuron_idx].m_activation;
            if (m_connections[i].m_weight > 0) // positive weight
            {
                double t_delta = (m_connections[i].m_hebb_rate
                                  * (t_max_weight - m_connections[i].m_weight)
                                  * t_incoming_neuron_activation
                                  * t_outgoing_neuron_activation)
                                 + m_connections[i].m_hebb_pre_rate * t_max_weight
                                   * t_incoming_neuron_activation
                                   * (t_outgoing_neuron_activation - 1.0);
                m_connections[i].m_weight = (m_connections[i].m_weight + t_delta);
            } else if (m_connections[i].m_weight < 0) // negative weight
            {
                // In the inhibatory case, we strengthen the synapse when output is low and
                // input is high
                double t_delta = m_connections[i].m_hebb_pre_rate
                                 * (t_max_weight - m_connections[i].m_weight)
                                 * t_incoming_neuron_activation
                                 * (1.0 - t_outgoing_neuron_activation)
                                 - m_connections[i].m_hebb_rate * t_max_weight
                                   * t_incoming_neuron_activation
                                   * t_outgoing_neuron_activation;
                m_connections[i].m_weight = -(m_connections[i].m_weight + t_delta);
            }

            Clamp(m_connections[i].m_weight, -a_Parameters.MaxWeight,
                  a_Parameters.MaxWeight);
        }
    }

    int NeuralNetwork::ConnectionExists(int a_to, int a_from) {
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            if ((m_connections[i].m_source_neuron_idx == a_from)
                && (m_connections[i].m_target_neuron_idx == a_to)) {
                return i;
            }
        }

        return -1;
    }

    void NeuralNetwork::RTRL_update_gradients() {
        // for every neuron
        for (unsigned int k = m_num_inputs; k < m_neurons.size(); k++) {
            // for all possible connections
            for (unsigned int i = m_num_inputs; i < m_neurons.size(); i++)
                // to
                for (unsigned int j = 0; j < m_neurons.size(); j++) // from
                {
                    int t_idx = ConnectionExists(i, j);
                    if (t_idx != -1) {
                        //double t_derivative = unsigned_sigmoid_derivative( m_neurons[k].m_activation );
                        double t_derivative = 0;
                        if (m_neurons[k].m_activation_function_type
                            == NEAT::UNSIGNED_SIGMOID) {
                            t_derivative = unsigned_sigmoid_derivative(
                                    m_neurons[k].m_activation);
                        } else if (m_neurons[k].m_activation_function_type
                                   == NEAT::TANH) {
                            t_derivative = tanh_derivative(
                                    m_neurons[k].m_activation);
                        }

                        double t_sum = 0;
                        // calculate the other sum
                        for (unsigned int l = 0; l < m_neurons.size(); l++) {
                            int t_l_idx = ConnectionExists(k, l);
                            if (t_l_idx != -1) {
                                t_sum += m_connections[t_l_idx].m_weight
                                         * m_neurons[l].m_sensitivity_matrix[i][j];
                            }
                        }

                        if (i == k) {
                            t_sum += m_neurons[j].m_activation;
                        }
                        m_neurons[k].m_sensitivity_matrix[i][j] = t_derivative
                                                                  * t_sum;
                    } else {
                        m_neurons[k].m_sensitivity_matrix[i][j] = 0;
                    }
                }

        }

    }

// please pay attention. notice here only one output is assumed
    void NeuralNetwork::RTRL_update_error(double a_target) {
        // add to total error
        m_total_error = (a_target - Output()[0]);
        // adjust each weight
        for (unsigned int i = 0; i < m_neurons.size(); i++) // to
        {
            for (unsigned int j = 0; j < m_neurons.size(); j++) // from
            {
                int t_idx = ConnectionExists(i, j);
                if (t_idx != -1) {
                    // we know the first output's index is m_num_inputs
                    double t_delta = m_total_error
                                     * m_neurons[m_num_inputs].m_sensitivity_matrix[i][j];
                    m_total_weight_change[t_idx] += t_delta * LEARNING_RATE;
                }
            }
        }
    }

    void NeuralNetwork::RTRL_update_weights() {
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            m_connections[i].m_weight += m_total_weight_change[i];
            m_total_weight_change[i] = 0; // clear this out
        }
        m_total_error = 0;
    }

    void NeuralNetwork::Save(const char *a_filename) {
        FILE *fil = fopen(a_filename, "w");
        Save(fil);
        fclose(fil);
    }

    void NeuralNetwork::Save(FILE *a_file) {
        fprintf(a_file, "NNstart\n");
        // save num inputs/outputs and stuff
        fprintf(a_file, "%d %d\n", m_num_inputs, m_num_outputs);
        // save neurons
        for (unsigned int i = 0; i < m_neurons.size(); i++) {
            // TYPE .. A .. B .. time_const .. bias .. activation_function_type .. split_y
            fprintf(a_file, "neuron %d %3.18f %3.18f %3.18f %3.18f %d %3.18f\n",
                    static_cast<int>(m_neurons[i].m_type), m_neurons[i].m_a,
                    m_neurons[i].m_b, m_neurons[i].m_timeconst, m_neurons[i].m_bias,
                    static_cast<int>(m_neurons[i].m_activation_function_type),
                    m_neurons[i].m_split_y);
        }
        // save connections
        for (unsigned int i = 0; i < m_connections.size(); i++) {
            // from .. to .. weight.. isrecur
            fprintf(a_file, "connection %d %d %3.18f %d %3.18f %3.18f\n",
                    m_connections[i].m_source_neuron_idx,
                    m_connections[i].m_target_neuron_idx, m_connections[i].m_weight,
                    static_cast<int>(m_connections[i].m_recur_flag),
                    m_connections[i].m_hebb_rate, m_connections[i].m_hebb_pre_rate);
        }
        // end
        fprintf(a_file, "NNend\n\n");
    }

    bool NeuralNetwork::Load(std::ifstream &a_DataFile) {
        std::string t_str;
        bool t_no_start = true, t_no_end = true;

        if (!a_DataFile) {
            ostringstream tStream;
            tStream << "NN file error!" << std::endl;
            //    throw NS::Exception(tStream.str());
        }

        // search for NNstart
        do {
            a_DataFile >> t_str;
            if (t_str == "NNstart")
                t_no_start = false;

        } while ((t_str != "NNstart") && (!a_DataFile.eof()));

        if (t_no_start)
            return false;

        Clear();

        // read in the input/output dimentions
        a_DataFile >> m_num_inputs;
        a_DataFile >> m_num_outputs;

        // read in all data
        do {
            a_DataFile >> t_str;

            // a neuron?
            if (t_str == "neuron") {
                Neuron t_n;

                // for type and aftype
                int t_type, t_aftype;

                a_DataFile >> t_type;
                a_DataFile >> t_n.m_a;
                a_DataFile >> t_n.m_b;
                a_DataFile >> t_n.m_timeconst;
                a_DataFile >> t_n.m_bias;
                a_DataFile >> t_aftype;
                a_DataFile >> t_n.m_split_y;

                t_n.m_type = static_cast<NEAT::NeuronType>(t_type);
                t_n.m_activation_function_type = static_cast<NEAT::ActivationFunction>(t_aftype);

                m_neurons.push_back(t_n);
            }

            // a connection?
            if (t_str == "connection") {
                Connection t_c;

                int t_isrecur;

                a_DataFile >> t_c.m_source_neuron_idx;
                a_DataFile >> t_c.m_target_neuron_idx;
                a_DataFile >> t_c.m_weight;
                a_DataFile >> t_isrecur;

                a_DataFile >> t_c.m_hebb_rate;
                a_DataFile >> t_c.m_hebb_pre_rate;

                t_c.m_recur_flag = static_cast<bool>(t_isrecur);

                m_connections.push_back(t_c);
            }


            if (t_str == "NNend")
                t_no_end = false;
        } while ((t_str != "NNend") && (!a_DataFile.eof()));

        if (t_no_end) {
            ostringstream tStream;
            tStream << "NNend not found in file!" << std::endl;
            //    throw NS::Exception(tStream.str());
        }

        return true;
    }

    bool NeuralNetwork::Load(const char *a_filename) {
        std::ifstream t_DataFile(a_filename);
        return Load(t_DataFile);
    }

}; // namespace NEAT
