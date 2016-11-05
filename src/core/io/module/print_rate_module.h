#ifndef print_rate_module_h
#define print_rate_module_h

#include "io/module/module.h"

class PrintRateModule : public Module {
    public:
        PrintRateModule(Layer *layer,
            std::string params);

        void report_output(Buffer *buffer);
        virtual IOType get_type() { return OUTPUT; }

    private:
        float *totals;
        int timesteps;
};

#endif
