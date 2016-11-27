#ifndef buffer_h
#define buffer_h

#include "util/constants.h"

class Buffer {
    public:
        Buffer(int input_size, int output_size, OutputType output_type);
        virtual ~Buffer();

        /* IO setters */
        void clear_input();
        void set_input(int offset, int size, float* source);
        void set_output(int offset, int size, Output* source);

        /* IO getters */
        float* get_input() { return this->input; }
        Output* get_output() { return this->output; }
        OutputType get_output_type() { return this->output_type; }

        OutputType output_type;
        int input_size;
        int output_size;

    private:
        float *input;
        Output *output;
};

#endif
