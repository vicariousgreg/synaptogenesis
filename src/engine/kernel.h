#ifndef kernel_h
#define kernel_h

#include "util/parallel.h"
#include "util/constants.h"

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
