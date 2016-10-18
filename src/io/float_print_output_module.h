#ifndef float_print_output_module_h
#define float_print_output_module_h

#include "io/output_module.h"

class FloatPrintOutputModule : public OutputModule {
    public:
        FloatPrintOutputModule(Layer *layer, std::string params);
        void report_output(Buffer *buffer);
};

#endif
