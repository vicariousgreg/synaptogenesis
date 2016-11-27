#ifndef kernel_data_h
#define kernel_data_h

#include "model/connection.h"
#include "state/state.h"
#include "util/parallel.h"

class KernelData;

/* Extractors are responsible for extracting values from output */
typedef float(*EXTRACTOR)(KernelData&, Output&);
void get_extractor(EXTRACTOR *dest, OutputType output_type);
DEVICE float extract_float(KernelData &kernel_data, Output &out);
DEVICE float extract_int(KernelData &kernel_data, Output &out);
DEVICE float extract_bit(KernelData &kernel_data, Output &out);

/* Data package that is passed into kernel functions */
class KernelData {
    public:
        KernelData(Connection *conn, State *state);

        bool convolutional;
        Opcode opcode;

        EXTRACTOR extractor;

        int overlap, stride;
        int fray;
        int delay;

        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;
        int num_weights;
        bool plastic;
        float max_weight;

        OutputType output_type;
        Output *outputs;
        float *inputs;
        float *weights;
};

#endif
