#ifndef one_hot_cyclic_input_module_h
#define one_hot_cyclic_input_module_h

#include "io/module.h"

class OneHotCyclicInputModule : public Module {
    public:
        OneHotCyclicInputModule(LayerList layers, ModuleConfig *config);

        void feed_input(Buffer *buffer);
        void cycle();

    private:
        int timesteps;
        int end;
        int cycle_rate;
        int index;
        float max_value;

    MODULE_MEMBERS
};

#endif