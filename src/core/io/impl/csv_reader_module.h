#ifndef csv_reader_module_h
#define csv_reader_module_h

#include "io/module.h"
#include "util/pointer.h"
#include "csvparser.h"

class CSVReaderModule : public Module {
    public:
        CSVReaderModule(Layer *layer, ModuleConfig *config);
        virtual ~CSVReaderModule();

        void cycle();

    protected:
        int age;
        int exposure;
        int curr_row;

        std::vector<Pointer<float>> data;

    MODULE_MEMBERS
};

#endif
