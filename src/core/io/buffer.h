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
        float* get_input() const { return input; }
        Output* get_output() const { return output; }

        const OutputType output_type;
        const int input_size;
        const int output_size;

    private:
        float *input;
        Output *output;
};

#endif
