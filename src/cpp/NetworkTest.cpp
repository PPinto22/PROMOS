#include <unordered_set>
#include <unordered_map>
#include "MultiNEAT/Genes.h"
#include "MultiNEAT/NeuralNetwork.h"

using namespace NEAT;

int main() {
    // Network structure: 2 inputs, 2 hidden nodes, 1 output
    // I1 --2- H1
    //   1\  /  1\
    //     \/     \ O1
    //     /\     /
    //   3/  \  1/
    // I2 --1- H2
    NeuralNetwork net;

    Neuron i1, i2, h1, h2, o1;
    h1.m_activation_function_type = UNSIGNED_SIGMOID;
    h2.m_activation_function_type = UNSIGNED_SIGMOID;
    o1.m_activation_function_type = UNSIGNED_SIGMOID;
    net.m_neurons.push_back(i1); // 0
    net.m_neurons.push_back(i2); // 1
    net.m_neurons.push_back(h1); // 2
    net.m_neurons.push_back(h2); // 3
    net.m_neurons.push_back(o1); // 4

    Connection ci1_h1, ci1_h2, ci2_h1, ci2_h2, ch1_o1, ch2_o1;
    ci1_h1.m_source_neuron_idx = 0;
    ci1_h1.m_target_neuron_idx = 2;
    ci1_h1.m_weight = 2;
    net.m_connections.push_back(ci1_h1);

    ci1_h2.m_source_neuron_idx = 0;
    ci1_h2.m_target_neuron_idx = 3;
    ci1_h2.m_weight = 1;
    net.m_connections.push_back(ci1_h2);

    ci2_h1.m_source_neuron_idx = 1;
    ci2_h1.m_target_neuron_idx = 2;
    ci2_h1.m_weight = 3;
    net.m_connections.push_back(ci2_h1);

    ci2_h2.m_source_neuron_idx = 1;
    ci2_h2.m_target_neuron_idx = 3;
    ci2_h2.m_weight = 1;
    net.m_connections.push_back(ci2_h2);

    ch1_o1.m_source_neuron_idx = 2;
    ch1_o1.m_target_neuron_idx = 4;
    ch1_o1.m_weight = 1;
    net.m_connections.push_back(ch1_o1);

    ch2_o1.m_source_neuron_idx = 3;
    ch2_o1.m_target_neuron_idx = 4;
    ch2_o1.m_weight = 1;
    net.m_connections.push_back(ch2_o1);

    net.m_num_inputs = 2;
    net.m_num_outputs = 1;

    net.Flush();
    std::vector<double> inputs{1, 1};
    net.Input(inputs);

    int depth = 2;
    for (int i = 0; i < depth; i++) {
        net.ActivateUseInternalBias();
        cout << "Depth " << i << ":" << endl;
        cout << "i1: " << net.m_neurons[0].m_activation << endl;
        cout << "i2: " << net.m_neurons[1].m_activation << endl;
        cout << "h1: " << net.m_neurons[2].m_activation << endl;
        cout << "h2: " << net.m_neurons[3].m_activation << endl;
        cout << "o1: " << net.m_neurons[4].m_activation << endl;
        cout << "output: " << net.Output()[0] << endl;
    }

    net.Flush();
    net.Input(inputs);
    cout << endl << "FeedForward: " << endl;
    net.FeedForward();
    cout << "i1: " << net.m_neurons[0].m_activation << endl;
    cout << "i2: " << net.m_neurons[1].m_activation << endl;
    cout << "h1: " << net.m_neurons[2].m_activation << endl;
    cout << "h2: " << net.m_neurons[3].m_activation << endl;
    cout << "o1: " << net.m_neurons[4].m_activation << endl;

    std::unordered_map<int, Connection *> t_toconnect;
    for (auto it = net.m_connections.begin(); it != net.m_connections.end(); it++) {
        long i = std::distance(net.m_connections.begin(), it);
        t_toconnect.insert(std::make_pair<int, Connection *>(i, &(*it)));
    }
    std::unordered_set<int> t_nodes_full_input;
    while (!t_toconnect.empty()) {
        long t_initial_size = t_toconnect.size();
        for (auto it = t_toconnect.begin(); it != t_toconnect.end(); /* no increment */) {
            int c_i = (*it).first;
            Connection& c = *((*it).second);
            Neuron& source = net.m_neurons[c.m_source_neuron_idx];
            Neuron& target = net.m_neurons[c.m_target_neuron_idx];
            // TODO: WIP
//            if (t_nodes_full_input.contains(c->m_source_neuron_idx))
//            {
//                c->m_signal = net.m_neurons[c->m_source_neuron_idx].m_activation * c->m_weight;
//                net.m_neurons[c->m_target_neuron_idx].m_activesum +=
//
////                m_neurons[m_connections[i].m_target_neuron_idx].m_activesum +=
////                        m_connections[i].m_signal;
//                it = t_toconnect.erase(it);
//            }
//            else
//            {
//                ++it;
//            }
        }
    }

    // Expected Output:
    //    Depth 0:
    //    i1: 1
    //    i2: 1
    //    h1: 0.993307
    //    h2: 0.880797
    //    o1: 0.5
    //    Depth 1:
    //    i1: 1
    //    i2: 1
    //    h1: 0.993307
    //    h2: 0.880797
    //    o1: 0.866932
}