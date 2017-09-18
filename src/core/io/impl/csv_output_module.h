#ifndef csv_output_module_h
#define csv_output_module_h

#include "io/module.h"
#include "util/pointer.h"

class CSVOutputModule : public Module {
    public:
        CSVOutputModule(Layer *layer, ModuleConfig *config);
        virtual ~CSVOutputModule();

        void report_output(Buffer *buffer, OutputType output_type);

    MODULE_MEMBERS
};

#endif
