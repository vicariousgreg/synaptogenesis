#ifdef __GUI__

#ifndef dsst_module_h
#define dsst_module_h

#include "io/module.h"

class DSSTWindow;

class DSSTModule : public Module {
    public:
        DSSTModule(LayerList layers, ModuleConfig *config);

        void feed_input_impl(Buffer *buffer);
        void report_output_impl(Buffer *buffer);

        void init();
        void update(Buffer *buffer);

        // Primarily used by DSST window
        int get_num_rows()      { return num_rows; }
        int get_num_columns()   { return num_cols; }
        int get_cell_rows()     { return cell_rows; }
        int get_cell_columns()  { return cell_cols; }
        int get_cell_size()     { return cell_size; }
        int get_spacing()       { return spacing; }
        int get_input_rows()    { return input_rows; }
        int get_input_columns() { return input_cols; }
        int get_input_size()    { return input_size; }

        // Useful for determining layer properties before construction
        static int get_num_rows(PropertyConfig *config);
        static int get_num_columns(PropertyConfig *config);
        static int get_cell_rows(PropertyConfig *config);
        static int get_cell_columns(PropertyConfig *config);
        static int get_cell_size(PropertyConfig *config);
        static int get_spacing(PropertyConfig *config);
        static int get_input_rows(PropertyConfig *config);
        static int get_input_columns(PropertyConfig *config);
        static int get_input_size(PropertyConfig *config);

    private:
        std::map<Layer*, std::string> params;
        Pointer<float> input_data;
        DSSTWindow *window;

        int num_cols, num_rows, cell_cols, cell_rows, cell_size, spacing;
        int input_rows, input_cols, input_size;

        void input_symbol(int index);

    MODULE_MEMBERS
};

#endif

#endif
