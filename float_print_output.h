#ifndef float_print_output_h
#define float_print_output_h

#include "output.h"

class FloatPrintOutput : public Output {
    public:
        FloatPrintOutput(Layer &layer, std::string params);
        void report_output(State *state);
};

#endif
