#ifndef frontend_h
#define frontend_h

#include <map>
#include <string>

#include "layer_info.h"

class GUI;
class GuiWindow;
class Layer;
class Environment;

class Frontend {
    public:
        Frontend();
        virtual ~Frontend();

        void set_window(GuiWindow *gui_window);

        virtual void init() { }
        virtual bool add_input_layer(Layer *layer,
            std::string params) = 0;
        virtual bool add_output_layer(Layer *layer,
            std::string params) = 0;
        virtual void update(Environment *environment) = 0;

        static const std::vector<Frontend*> get_instances();
        static void init_all();
        static void launch_all();
        static void update_all(Environment *environment);
        static void cleanup();

    protected:
        GUI *gui;
        GuiWindow *gui_window;
        static std::vector<Frontend*> instances;
        std::map<Layer*, LayerInfo*> layer_map;
};

#endif
