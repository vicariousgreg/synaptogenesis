#ifndef kernel_h
#define kernel_h

#include "util/parallel.h"
#include "util/constants.h"

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

class Instruction;

/* Clears input data */
GLOBAL void clear_data(float* data, int count);

/* Extractors are responsible for extracting values from output */
typedef float(*EXTRACTOR)(Instruction&, Output&);
void get_extractor(EXTRACTOR *dest, OutputType output_type);
DEVICE float extract_float(Instruction &inst, Output &out);
DEVICE float extract_int(Instruction &inst, Output &out);
DEVICE float extract_bit(Instruction &inst, Output &out);

/* Kernels are responsible for performing connection computation */
typedef void(*KERNEL)(Instruction);
void get_kernel(KERNEL *dest, ConnectionType conn_type);
GLOBAL void calc_fully_connected(Instruction inst);
GLOBAL void calc_one_to_one(Instruction inst);
GLOBAL void calc_divergent(Instruction inst);
GLOBAL void calc_convergent(Instruction inst);

#endif
