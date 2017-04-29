#ifndef one_hot_cyclic_input_module_h
#define one_hot_cyclic_input_module_h

#include "io/module/module.h"

class OneHotCyclicInputModule : public Module {
    public:
        OneHotCyclicInputModule(Layer *layer, std::string params);

        void feed_input(Buffer *buffer);
        virtual IOTypeMask get_type() { return INPUT; }

    private:
        void print();
        void cycle(Buffer *buffer);
        void clear(Buffer *buffer);

        int timesteps;
        int end;
        int cycle_rate;
        int index;
        float max_value;
};

#endif
