#ifndef frontend_h
#define frontend_h

#include <vector>
#include <map>
#include <string>

#include "layer_info.h"

class GUI;
class GuiWindow;
class Layer;
class Buffer;

class Frontend {
    public:
        Frontend();
        virtual ~Frontend();

        void set_window(GuiWindow *gui_window);
        void add_layer(Layer* layer, LayerInfo* info);

        virtual void init() { }
        virtual bool add_input_layer(Layer *layer,
            std::string params) = 0;
        virtual bool add_output_layer(Layer *layer,
            std::string params) = 0;
        virtual void update(Buffer *buffer) = 0;

        virtual std::string get_name() = 0;

        static Frontend* get_instance(std::string name);
        static void init_all();
        static void launch_all();
        static void update_all(Buffer *buffer);
        static void quit();
        static void cleanup();

    protected:
        static std::vector<Frontend*> instances;

        GUI *gui;
        GuiWindow *gui_window;
        std::vector<Layer*> layer_list;
        std::map<Layer*, LayerInfo*> layer_map;
};

#endif
