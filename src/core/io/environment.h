#ifndef environment_h
#define environment_h

#include <map>

#include "model/structure.h"
#include "io/environment_model.h"
#include "io/module/module.h"
#include "io/buffer.h"

class Model;

class Environment {
    public:
        Environment(EnvironmentModel *env_model, Model* net_model,
            bool suppress_output=false);
        virtual ~Environment();

        void step_input();
        void step_output();
        void ui_init();
        void ui_launch();
        void ui_update();

        Buffer* get_buffer() { return buffer; }
        IOTypeMask get_io_type(Layer *layer) { return io_types[layer]; }
        bool is_input(Layer *layer) { return get_io_type(layer) & INPUT; }
        bool is_output(Layer *layer) { return get_io_type(layer) & OUTPUT; }
        bool is_expected(Layer *layer) { return get_io_type(layer) & EXPECTED; }

    private:
        std::map<Layer*, IOTypeMask> io_types;
        ModuleList all_modules, input_modules, expected_modules, output_modules;
        Buffer* buffer;
};

#endif
