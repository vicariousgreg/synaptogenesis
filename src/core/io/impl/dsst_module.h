#ifndef dsst_module_h
#define dsst_module_h

#include "io/module.h"

class DSSTWindow;

class DSSTModule : public Module {
    public:
        DSSTModule(LayerList layers, ModuleConfig *config);

        void feed_input(Buffer *buffer);
        void report_output(Buffer *buffer);

        void init();
        void update(Buffer *buffer);

        void input_symbol(int index);

        static const int num_cols = 18;
        static const int num_rows = 8;
        static const int cell_cols = 8;
        static const int cell_rows = 1+2*DSSTModule::cell_cols;
        static const int cell_size = DSSTModule::cell_rows * DSSTModule::cell_cols;
        static const int spacing = DSSTModule::cell_cols/4;

        static const int input_rows =
            (DSSTModule::num_rows + 2)
            * (DSSTModule::cell_rows + DSSTModule::spacing)
            - DSSTModule::spacing;
        static const int input_columns =
            DSSTModule::num_cols
            * (DSSTModule::spacing + DSSTModule::cell_cols)
            - DSSTModule::spacing;
        static const int input_size =input_rows * input_columns;

    private:
        std::map<Layer*, std::string> params;

        Pointer<float> input_data;

        bool ui_dirty;
        bool input_dirty;
        DSSTWindow *window;

    MODULE_MEMBERS
};

#endif
