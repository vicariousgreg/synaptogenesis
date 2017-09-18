#ifndef csv_evaluator_module_h
#define csv_evaluator_module_h

#include "io/impl/csv_expected_module.h"
#include "util/pointer.h"
#include "csvparser.h"

class CSVEvaluatorModule : public CSVExpectedModule {
    public:
        CSVEvaluatorModule(Layer *layer, ModuleConfig *config);

        void report_output(Buffer *buffer);

    private:
        int correct;
        float total_SSE;

    MODULE_MEMBERS
};

#endif
