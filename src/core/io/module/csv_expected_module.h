#ifndef csv_expected_module_h
#define csv_expected_module_h

#include "io/module/module.h"
#include "util/pointer.h"
#include "csvparser.h"

class CSVExpectedModule : public Module {
    public:
        CSVExpectedModule(Layer *layer, ModuleConfig *config);
        virtual ~CSVExpectedModule();

        void feed_expected(Buffer *buffer);
        virtual IOTypeMask get_type() { return EXPECTED; }

    private:
        int age;
        int exposure;
        int curr_row;

        std::vector<Pointer<Output> > data;

    MODULE_MEMBERS
};

#endif
