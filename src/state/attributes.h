#ifndef attributes_h
#define attributes_h

#include "model/model.h"
#include "io/buffer.h"
#include "constants.h"

class Attributes {
    public:
        Attributes(Model *model, OutputType output_type);
        virtual ~Attributes();

#ifdef PARALLEL
        void get_input_from(Buffer *buffer, cudaStream_t stream);
        void send_output_to(Buffer *buffer, cudaStream_t stream);
#else
        void get_input_from(Buffer *buffer);
        void send_output_to(Buffer *buffer);
#endif

        // Number of neurons
        int total_neurons;
        int num_neurons[IO_TYPE_SIZE];
        int start_index[IO_TYPE_SIZE];

        // Neuron input
        float* input;

        // Neuron output
        OutputType output_type;
        Output* output;
        Output* recent_output;
};

#endif
