#ifndef csv_module_h
#define csv_module_h

#include "io/module.h"
#include "util/pointer.h"

class CSVReaderModule : public Module {
    public:
        CSVReaderModule(LayerList layers, ModuleConfig *config);
        virtual ~CSVReaderModule();

        void cycle();

    protected:
        std::string filename;
        bool verbose;
        int age;
        int exposure;
        int curr_row;

        std::vector<Pointer<float>> data;

    MODULE_MEMBERS
};

class CSVInputModule : public CSVReaderModule {
    public:
        CSVInputModule(LayerList layers, ModuleConfig *config);

        void feed_input(Buffer *buffer);

    MODULE_MEMBERS
};

class CSVExpectedModule : public CSVReaderModule {
    public:
        CSVExpectedModule(LayerList layers, ModuleConfig *config);

        void feed_expected(Buffer *buffer);

    MODULE_MEMBERS
};

class CSVOutputModule : public Module {
    public:
        CSVOutputModule(LayerList layers, ModuleConfig *config);
        virtual ~CSVOutputModule();

        void report_output(Buffer *buffer);

    MODULE_MEMBERS
};

class CSVEvaluatorModule : public CSVExpectedModule {
    public:
        CSVEvaluatorModule(LayerList layers, ModuleConfig *config);

        void report_output(Buffer *buffer);
        void report(Report *report);

    private:
        std::map<Layer*, int> correct;
        std::map<Layer*, float> total_SSE;

    MODULE_MEMBERS
};

#endif
