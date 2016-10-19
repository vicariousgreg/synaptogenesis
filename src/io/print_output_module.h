#ifndef print_output_module_h
#define print_output_module_h

#include "io/module.h"

class PrintOutputModule : public Module {
    public:
        PrintOutputModule(Layer *layer,
            std::string params, std::string &driver_type);
        void report_output(Buffer *buffer);
        virtual IOType get_type() { return OUTPUT; }

    private:
        int history_length;
        int step_size;
        int counter;
        int refresh_rate;
        unsigned int maximum;
        unsigned int* reverses;
};

#endif
