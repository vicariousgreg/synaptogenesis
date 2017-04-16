#include "io/module/module.h"
#include "io/module/print_output_module.h"
#include "io/module/print_rate_module.h"
#include "io/module/random_input_module.h"
#include "io/module/one_hot_random_input_module.h"
#include "io/module/image_input_module.h"
#include "io/module/csv_input_module.h"
#include "io/module/csv_output_module.h"
#include "io/module/visualizer_input_module.h"
#include "io/module/visualizer_output_module.h"
#include "io/module/maze_input_module.h"
#include "io/module/maze_output_module.h"
#include "io/module/dummy_input_module.h"
#include "io/module/dummy_output_module.h"
#include "util/error_manager.h"

Module* build_module(Layer *layer, std::string type,
        std::string params) {
    if (type == "random_input")
        return new RandomInputModule(layer, params);
    else if (type == "one_hot_random_input")
        return new OneHotRandomInputModule(layer, params);
    else if (type == "image_input")
        return new ImageInputModule(layer, params);
    else if (type == "csv_input")
        return new CSVInputModule(layer, params);
    else if (type == "csv_output")
        return new CSVOutputModule(layer, params);
    else if (type == "print_output")
        return new PrintOutputModule(layer, params);
    else if (type == "print_rate")
        return new PrintRateModule(layer, params);
    else if (type == "visualizer_input")
        return new VisualizerInputModule(layer, params);
    else if (type == "visualizer_output")
        return new VisualizerOutputModule(layer, params);
    else if (type == "maze_input")
        return new MazeInputModule(layer, params);
    else if (type == "maze_output")
        return new MazeOutputModule(layer, params);
    else if (type == "dummy_input")
        return new DummyInputModule(layer, params);
    else if (type == "dummy_output")
        return new DummyOutputModule(layer, params);
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized module!");
}
