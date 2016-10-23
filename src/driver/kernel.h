#ifndef kernel_h
#define kernel_h

#include "driver/instruction.h"

GLOBAL void clear_data(float* data, int count);

GLOBAL void calc_fully_connected(Instruction inst);

GLOBAL void calc_one_to_one(Instruction inst);

GLOBAL void calc_divergent(Instruction inst);

GLOBAL void calc_convergent(Instruction inst);

#endif
