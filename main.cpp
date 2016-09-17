#include <iostream>
#include <vector>

#include "environment.h"

void print_values(Environment env) {
    for (int nid = 0 ; nid < env.num_neurons ; ++nid) {
        double current = ((double*)env.nat[CURRENT])[nid];
        double voltage = ((double*)env.nat[VOLTAGE])[nid];
        NeuronParameters &params = ((NeuronParameters*)env.nat[PARAMS])[nid];
        //cout << env.spikes[nid];
        //cout << current << " " << voltage << " " << a << " " << b << " " << c << " " << d;;
        //cout << "\n";
    }
}
void print_spikes(Environment env) {
    int* spikes = env.get_spikes();
    for (int nid = 0 ; nid < env.num_neurons ; ++nid) {
        char c = spikes[nid] ? '*' : ' ';
        cout << c;
    }
    cout << "|\n";
}

void print_currents(Environment env) {
    double* currents = env.get_currents();
    for (int nid = 0 ; nid < env.num_neurons ; ++nid) {
        cout << currents[nid] << " ";
    }
    cout << "|\n";
}

int main(void) {
    Environment env;
    int size = 800;
    int iterations = 1000;

    int pos = env.add_randomized_layer(size, 1);
    int neg = env.add_randomized_layer(size / 4, -1);
    env.connect_layers(pos, pos, true, .5);
    env.connect_layers(pos, neg, true, .5);
    env.connect_layers(neg, pos, true, 1);
    env.connect_layers(neg, neg, true, 1);
    env.build();

    for (int i = 0 ; i < iterations ; ++i) {
        //print_values(env);
        print_spikes(env);
        print_currents(env);
        env.inject_random_current(pos, 5);
        env.inject_random_current(neg, 2);
        env.cycle();
    }

    return 0;
}
