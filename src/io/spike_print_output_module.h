#ifndef spike_print_output_module_h
#define spike_print_output_module_h

#include "io/output_module.h"

class SpikePrintOutputModule : public OutputModule {
    public:
        SpikePrintOutputModule(Layer *layer, std::string params);
        void report_output(Buffer *buffer);

    int history_length;
    int step_size;
    int counter;
    int refresh_rate;
    unsigned int maximum;
    unsigned int* reverses;
};

#endif
