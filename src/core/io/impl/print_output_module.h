#ifndef print_output_module_h
#define print_output_module_h

#include "io/module.h"

class PrintOutputModule : public Module {
    public:
        PrintOutputModule(LayerList layers, ModuleConfig *config);

        void report_output_impl(Buffer *buffer);

    private:
        int history_length;
        int step_size;
        int counter;
        int refresh_rate;
        unsigned int maximum;
        unsigned int shift;

    MODULE_MEMBERS
};

#endif
