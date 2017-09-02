#ifndef dsst_input_module_h
#define dsst_input_module_h

#include "io/module/module.h"
#include "dsst.h"

class DSSTInputModule : public Module {
    public:
        DSSTInputModule(Layer *layer, ModuleConfig *config);

        void feed_input(Buffer *buffer);
        virtual IOTypeMask get_type() { return INPUT; }

    private:
        DSST* dsst;
        std::string params;

    MODULE_MEMBERS
};

#endif
