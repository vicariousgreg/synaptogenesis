#ifndef buffer_h
#define buffer_h

#include "state/state.h"

class Buffer {
    public:
        Buffer(int num_neurons, int output_size);
        ~Buffer() {
            free(this->input);
            free(this->output);
        }

        void clear_input();
        void set_input(int offset, int size, float* source);

        void send_input_to(State *state);
        void retrieve_output_from(State *state);

        float* get_input();
        void* get_output();

    private:
        int output_size;
        int num_neurons;
        float *input;
        void *output;
};

#endif
