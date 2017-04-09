#ifndef visualizer_h
#define visualizer_h

#include "frontend.h"

class VisualizerWindow;
class Layer;
class Environment;

class Visualizer : public Frontend {
    public:
        static Visualizer *get_instance(bool init);

        virtual ~Visualizer();

        bool add_input_layer(Layer *layer, std::string params);
        bool add_output_layer(Layer *layer, std::string params);
        void update(Environment *environment);

    private:
        static int instance_id;
        Visualizer();

        VisualizerWindow *visualizer_window;
};

#endif
