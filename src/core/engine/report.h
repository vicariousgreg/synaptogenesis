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

class Report {
    public:
        Report(Engine* engine, State* state, int iterations, float total_time);
        virtual ~Report();

        void print();
        void add_report(Module *module, Layer *layer, PropertyConfig *props);

        const int iterations;
        const float total_time;
        const float average_time;
        const size_t network_bytes;
        const size_t state_buffer_bytes;
        const size_t engine_buffer_bytes;

        class LayerReport {
            public:
                LayerReport(std::string module, std::string structure,
                    std::string layer, PropertyConfig *properties)
                        : module(module),
                          structure(structure),
                          layer(layer),
                          properties(properties) { }
                virtual ~LayerReport() { delete properties; }

                const std::string module, structure, layer;
                PropertyConfig * const properties;
        };

    protected:
        std::map<Layer*, std::vector<Report::LayerReport*>> layer_reports;
};

#endif
