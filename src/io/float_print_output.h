#ifndef float_print_output_h
#define float_print_output_h

#include "io/output.h"

class FloatPrintOutput : public Output {
    public:
        FloatPrintOutput(Layer *layer, std::string params);
        void report_output(Buffer *buffer);
};

#endif
