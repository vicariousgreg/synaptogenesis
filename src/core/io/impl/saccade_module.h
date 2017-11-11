#ifndef saccade_module_h
#define saccade_module_h

#include "io/module.h"

class SaccadeWindow;

class SaccadeModule : public Module {
    public:
        SaccadeModule(LayerList layers, ModuleConfig *config);

        void feed_input_impl(Buffer *buffer);
        void report_output_impl(Buffer *buffer);

        void init();
        void update(Buffer *buffer);

        // Primarily used by Saccade window
        static int get_input_rows()    { return 100; }
        static int get_input_columns() { return 200; }
        static int get_input_size()    { return 100 * 200; }

    private:
        std::map<Layer*, std::string> params;
        Pointer<float> input_data;
        SaccadeWindow *window;

    MODULE_MEMBERS
};

#endif
