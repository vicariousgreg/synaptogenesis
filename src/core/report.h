#ifndef report_h
#define report_h

#include <cstddef>
#include <map>
#include <vector>

#include "util/property_config.h"

class Layer;
class State;
class Engine;
class Module;

class Report : public PropertyConfig {
    public:
        Report(Engine* engine, State* state,
            size_t iterations, float total_time);

        void print();
        void add_report(Module *module, Layer *layer, PropertyConfig props);

    protected:
        std::map<Layer*, std::vector<int>> layer_indices;
};

#endif
