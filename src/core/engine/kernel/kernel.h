#ifndef kernel_h
#define kernel_h

#include "util/parallel.h"
#include "util/constants.h"

#define MOD_RATE 0.3
#define MOD_DECAY 0.01
#define MOD_MAX 10.0
#define SUM_COEFFICIENT 0.5
#define WEIGHT_DECAY 0.025

/* Synaptic operations
 * |prior| is the current state of the neuron.
 * |input| is the synaptic input accomulated from one connection.
 *
 * ADD represent traditional excitatory input
 * SUB represent traditional inhibitory input
 * MULT and DIV represent modulatory input that can be used for gating
 * */
inline DEVICE float calc(Opcode opcode, float prior, float input) {
    switch (opcode) {
        case ADD:  return prior + input;
        case SUB:  return prior - input;
        case MULT: return prior * (1+input);
        case DIV:  return prior / (1+input);
    }
    return 0.0;
}

class KernelData;

/* Clears input data */
GLOBAL void clear_data(float* data, int count);

/* Extractors are responsible for extracting values from output */
typedef float(*EXTRACTOR)(KernelData&, Output&);
void get_extractor(EXTRACTOR *dest, OutputType output_type);
DEVICE float extract_float(KernelData &kernel_data, Output &out);
DEVICE float extract_int(KernelData &kernel_data, Output &out);
DEVICE float extract_bit(KernelData &kernel_data, Output &out);

#endif
