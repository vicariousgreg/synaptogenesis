#ifndef one_hot_random_input_module_h
#define one_hot_random_input_module_h

#include "io/module.h"

class OneHotRandomInputModule : public Module {
    public:
        OneHotRandomInputModule(LayerList layers, ModuleConfig *config);
        virtual ~OneHotRandomInputModule();

        void feed_input(Buffer *buffer);
        void cycle();

    private:
        bool verbose;
        int timesteps;
        int end;
        int shuffle_rate;
        float max_value;
        float *random_values;

    MODULE_MEMBERS
};

#endif
