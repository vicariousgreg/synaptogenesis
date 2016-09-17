#include <cstdlib>

void dot(int sign, int* spikes, double* weights, double* currents,
         int from_size, int to_index) {
    for (int index = 0 ; index < from_size ; ++index) {
        currents[to_index] += sign * spikes[index] * weights[index];
    }
}

void mult(int sign, int* spikes, double* weights, double* currents,
          int from_size, int to_size) {
    for (int row = 0 ; row < to_size ; ++row) {
        dot(sign, spikes, weights, currents, from_size, row);
    }
}
