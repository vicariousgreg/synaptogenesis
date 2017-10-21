#ifndef print_rate_module_h
#define print_rate_module_h

#include "io/module.h"

class PrintRateModule : public Module {
    public:
        PrintRateModule(LayerList layers,
            ModuleConfig *config);

        void report_output_impl(Buffer *buffer);
        void cycle_impl();

    private:
        int window;
        float *totals;
        int timesteps;

    MODULE_MEMBERS
};

#endif
