#ifndef one_hot_cyclic_input_module_h
#define one_hot_cyclic_input_module_h

#include "io/module/module.h"

class OneHotCyclicInputModule : public Module {
    public:
        OneHotCyclicInputModule(Layer *layer, std::string params);
        virtual ~OneHotCyclicInputModule();

        void feed_input(Buffer *buffer);
        virtual IOTypeMask get_type() { return INPUT; }

    private:
        void cycle();

        int timesteps;
        int end;
        int cycle_rate;
        int index;
        float max_value;
        float *vals;
};

#endif
