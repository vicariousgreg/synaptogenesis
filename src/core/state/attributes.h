#ifndef attributes_h
#define attributes_h

#include "model/model.h"
#include "io/buffer.h"
#include "util/constants.h"

class Attributes {
    public:
        Attributes(Model *model, OutputType output_type);
        virtual ~Attributes();

        float* get_input() { return input; }
        OutputType get_output_type() { return output_type; }
        Output* get_recent_output() { return recent_output; }
        Output* get_output(int word_index = 0) {
            return output + (total_neurons * word_index);
        }

        int get_num_neurons() { return total_neurons; }
        int get_num_neurons(IOType type) { return num_neurons[type]; }
        int get_start_index(IOType type) { return start_index[type]; }

    protected:
        friend class State;
#ifdef PARALLEL
        virtual void update(int start_index, int count, cudaStream_t &stream) = 0;
        void get_input_from(Buffer *buffer, cudaStream_t &stream);
        void send_output_to(Buffer *buffer, cudaStream_t &stream);
#else
        virtual void update(int start_index, int count) = 0;
        void get_input_from(Buffer *buffer);
        void send_output_to(Buffer *buffer);
#endif

        // Number of neurons
        int total_neurons;
        int num_neurons[IO_TYPE_SIZE];

        // Start indices by type
        int start_index[IO_TYPE_SIZE];

        // Neuron input
        float* input;

        // Neuron output
        OutputType output_type;
        Output* output;
        Output* recent_output;
};

Attributes *build_attributes(Model *model);

#endif
