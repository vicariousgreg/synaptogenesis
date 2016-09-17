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
    for (int nid = 0 ; nid < env.num_neurons ; ++nid) {
        char c = ((int*)env.nat[SPIKE])[nid] ? '*' : ' ';
        cout << c;
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
        env.set_random_currents(pos, 5);
        env.set_random_currents(neg, 2);
        env.cycle();
    }

    return 0;
}
