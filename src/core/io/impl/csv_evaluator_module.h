#ifndef csv_evaluator_module_h
#define csv_evaluator_module_h

#include "io/impl/csv_expected_module.h"
#include "util/pointer.h"
#include "csvparser.h"

class CSVEvaluatorModule : public CSVExpectedModule {
    public:
        CSVEvaluatorModule(LayerList layers, ModuleConfig *config);

        void report_output(Buffer *buffer);

    private:
        std::map<Layer*, int> correct;
        std::map<Layer*, float> total_SSE;

    MODULE_MEMBERS
};

#endif
