#ifndef instruction_h
#define instruction_h

#include "model/connection.h"

class Instruction {
    public:
        Instruction(Connection *conn, Output *outputs,
            OutputType output_type, float *inputs, float *weights) :
                type(conn->type),
                convolutional(conn->convolutional),
                opcode(conn->opcode),
                overlap(conn->overlap),
                stride(conn->stride),
                delay(conn->delay),
                from_size(conn->from_layer->size),
                from_rows(conn->from_layer->rows),
                from_columns(conn->from_layer->columns),
                to_size(conn->to_layer->size),
                to_rows(conn->to_layer->rows),
                to_columns(conn->to_layer->columns),
                num_weights(conn->num_weights),
                outputs(outputs + conn->from_layer->index),
                output_type(output_type),
                inputs(inputs + conn->to_layer->index),
                weights(weights) {
            this->fray = 
                (to_rows == from_rows and to_columns == from_columns)
                    ? overlap / 2 : 0;
        }

        ConnectionType type;
        bool convolutional;
        Opcode opcode;

        int overlap, stride;
        int fray;
        int delay;

        int from_size, from_rows, from_columns;
        int to_size, to_rows, to_columns;
        int num_weights;

        OutputType output_type;
        Output *outputs;
        float *inputs;
        float *weights;
};

#endif
