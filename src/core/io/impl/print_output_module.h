#ifndef print_output_module_h
#define print_output_module_h

#include "io/module.h"

class PrintOutputModule : public Module {
    public:
        PrintOutputModule(Layer *layer, ModuleConfig *config);

        void report_output(Buffer *buffer, OutputType output_type);

    private:
        int history_length;
        int step_size;
        int counter;
        int refresh_rate;
        unsigned int maximum;
        unsigned int shift;
        OutputType output_type;

    MODULE_MEMBERS
};

#endif
