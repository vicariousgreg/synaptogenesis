#ifndef visualizer_h
#define visualizer_h

#include "frontend.h"

class VisualizerWindow;
class Layer;
class Environment;

class Visualizer : public Frontend {
    public:
        static Visualizer *get_instance(bool init);

        virtual bool add_input_layer(Layer *layer, std::string params);
        virtual bool add_output_layer(Layer *layer, std::string params);
        virtual void update(Environment *environment);

    protected:
        static int instance_id;
        Visualizer();

        VisualizerWindow *visualizer_window;
};

#endif
