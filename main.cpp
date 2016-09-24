#include <iostream>
#include <cstdio>

#include "environment.h"
#include "state.h"
#include "model.h"
#include "weight_matrix.h"
#include "tools.h"
#include "parallel.h"

/* Prints a line for a timestep containing markers for neuron spikes.
 * If a neuron spikes, an asterisk will be printed.  Otherwise, a space */
void print_spikes(Environment env) {
    int* spikes = env.get_spikes();
    for (int nid = 0 ; nid < env.model.num_neurons ; ++nid) {
        char c = (spikes[nid] % 2) ? '*' : ' ';
        std::cout << c;
    }
    std::cout << "|\n";
}

/* Prints a line for a timestep containing neuron currents */
void print_currents(Environment env) {
    float* currents = env.get_current();
    for (int nid = 0 ; nid < env.model.num_neurons ; ++nid) {
        std::cout << currents[nid] << " ";
    }
    std::cout << "|\n";
}

int main(void) {
    Model model;
    int size = 800 * 1;
    int iterations = 500;

    int pos = model.add_randomized_layer(size, 1);
    int neg = model.add_randomized_layer(size / 4, -1);
    model.connect_layers(pos, pos, true, .5, FULLY_CONNECTED);
    model.connect_layers(pos, neg, true, .5, FULLY_CONNECTED);
    model.connect_layers(neg, pos, true, 1, FULLY_CONNECTED);
    model.connect_layers(neg, neg, true, 1, FULLY_CONNECTED);

    try {
        // Start timer
        timer.start();
        Environment env(model);
        timer.stop("Initialization");

        timer.start();

        for (int i = 0 ; i < iterations ; ++i) {
            //print_values(env);
            print_spikes(env);
            //print_currents(env);
            env.inject_random_current(pos, 5);
            env.inject_random_current(neg, 2);
            env.timestep();
        }

        float time = timer.stop("Total time");
        printf("Time averaged over %d iterations: %f\n", iterations, time/iterations);
    } catch (const char* msg) { 
        printf("%s\n", msg);
        return 1;
    }

#ifdef PARALLEL
    check_memory();
#endif

    return 0;
}
