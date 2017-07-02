#ifndef print_output_module_h
#define print_output_module_h

#include "io/module/module.h"

class PrintOutputModule : public Module {
    public:
        PrintOutputModule(Layer *layer, ModuleConfig *config);

        void report_output(Buffer *buffer, OutputType output_type);
        virtual IOTypeMask get_type() { return OUTPUT; }

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
