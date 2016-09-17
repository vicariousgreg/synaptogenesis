#ifndef parallel_h
#define parallel_h

void dot(int sign, int* spikes, double* weights, double* currents,
         int from_size, int to_index);

void mult(int sign, int* spikes, double* weights, double* currents,
          int from_size, int to_size);

#endif
