#ifndef print_output_module_h
#define print_output_module_h

#include "io/module/module.h"

class PrintOutputModule : public Module {
    public:
        PrintOutputModule(Layer *layer,
            std::string params);
        virtual ~PrintOutputModule() {
            free(this->reverses);
        }

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
