#ifndef print_rate_module_h
#define print_rate_module_h

#include "io/module/module.h"

class PrintRateModule : public Module {
    public:
        PrintRateModule(Layer *layer,
            std::string params);

        void report_output(Buffer *buffer, OutputType output_type);
        virtual IOType get_type() { return OUTPUT; }

    private:
        int rate;
        float *totals;
        int timesteps;
};

#endif
