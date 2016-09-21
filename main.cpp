#include <iostream>
#include <vector>
#include <cstdio>

#include "environment.h"
#include "tools.h"

#ifdef parallel
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/** Checks cuda memory usage and availability */
void check_memory() {
    // show memory usage of GPU
    size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}
#endif

/* Prints a line for a timestep containing markers for neuron spikes.
 * If a neuron spikes, an asterisk will be printed.  Otherwise, a space */
void print_spikes(Environment env) {
    int* spikes = env.get_spikes();
    for (int nid = 0 ; nid < env.num_neurons ; ++nid) {
        char c = (spikes[nid] % 2) ? '*' : ' ';
        cout << c;
    }
    cout << "|\n";
}

/* Prints a line for a timestep containing neuron currents */
void print_currents(Environment env) {
    float* currents = env.get_currents();
    for (int nid = 0 ; nid < env.num_neurons ; ++nid) {
        cout << currents[nid] << " ";
    }
    cout << "|\n";
}

int main(void) {
    // Start timer
    timer.start();

    Environment env;
    int size = 800 * 21;
    int iterations = 50;

    int pos = env.add_randomized_layer(size, 1);
    int neg = env.add_randomized_layer(size / 4, -1);
    env.connect_layers(pos, pos, true, .5);
    env.connect_layers(pos, neg, true, .5);
    env.connect_layers(neg, pos, true, 1);
    env.connect_layers(neg, neg, true, 1);
    env.build();

    timer.stop("Initialization");

    timer.start();

    for (int i = 0 ; i < iterations ; ++i) {
        //print_values(env);
        //print_spikes(env);
        //print_currents(env);
        env.inject_random_current(pos, 5);
        env.inject_random_current(neg, 2);
        env.cycle();
    }

    float time = timer.stop("Total time");
    printf("Time averaged over %d iterations: %f\n", iterations, time/iterations);

#ifdef parallel
    check_memory();
#endif

    return 0;
}
