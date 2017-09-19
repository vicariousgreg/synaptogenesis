#ifndef csv_expected_module_h
#define csv_expected_module_h

#include "io/impl/csv_reader_module.h"
#include "util/pointer.h"
#include "csvparser.h"

class CSVExpectedModule : public CSVReaderModule {
    public:
        CSVExpectedModule(LayerList layers, ModuleConfig *config);

        void feed_expected(Buffer *buffer);

    MODULE_MEMBERS
};

#endif
