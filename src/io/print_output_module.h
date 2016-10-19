#ifndef print_output_module_h
#define print_output_module_h

#include "io/output_module.h"

class PrintOutputModule : public OutputModule {
    public:
        PrintOutputModule(Layer *layer,
            std::string params, std::string &driver_type);
        void report_output(Buffer *buffer);

    int history_length;
    int step_size;
    int counter;
    int refresh_rate;
    unsigned int maximum;
    unsigned int* reverses;
};

#endif
