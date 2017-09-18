#ifndef print_rate_module_h
#define print_rate_module_h

#include "io/module.h"

class PrintRateModule : public Module {
    public:
        PrintRateModule(Layer *layer,
            ModuleConfig *config);

        void report_output(Buffer *buffer);
        void cycle();

    private:
        int rate;
        float *totals;
        int timesteps;

    MODULE_MEMBERS
};

#endif
