#ifndef constants_h
#define constants_h

/* Voltage threshold for neuron spiking. */
#define SPIKE_THRESH 30

/* Euler resolution for voltage update. */
#define EULER_RES 10

/* Size (in integers) of spike history bit vector.
 * A size of 1 indicates the usage of 1 integer, which on most systems is
 *     4 bytes, or 32 bits.  This means that a history of 32 timesteps will
 *     be kept for neuron spikes.
 * The history size limits the longest connection delay possible, and also
 *     affects learning for STDP rules that use more than just the most
 *     recent spike.
 */
#define HISTORY_SIZE 8

#endif
