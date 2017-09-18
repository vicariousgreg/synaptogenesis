#ifndef csv_input_module_h
#define csv_input_module_h

#include "io/impl/csv_reader_module.h"

class CSVInputModule : public CSVReaderModule {
    public:
        CSVInputModule(Layer *layer, ModuleConfig *config);

        void feed_input(Buffer *buffer);

    MODULE_MEMBERS
};

#endif
