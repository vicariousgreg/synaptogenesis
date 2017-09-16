#ifndef one_step_input_module_h
#define one_step_input_module_h

#include "io/module/module.h"

class OneStepInputModule : public Module {
    public:
        OneStepInputModule(Layer *layer, ModuleConfig *config);
        virtual ~OneStepInputModule();

        void feed_input(Buffer *buffer);
        virtual IOTypeMask get_type() { return INPUT; }

    private:
        float *random_values;
        bool active;
        bool cleared;
        bool verbose;

    MODULE_MEMBERS
};

#endif
