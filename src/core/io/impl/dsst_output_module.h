#ifndef dsst_output_module_h
#define dsst_output_module_h

#include "io/module.h"
#include "dsst.h"

class DSSTOutputModule : public Module {
    public:
        DSSTOutputModule(LayerList layers, ModuleConfig *config);

        void report_output(Buffer *buffer);

    private:
        DSST* dsst;

    MODULE_MEMBERS
};

#endif
