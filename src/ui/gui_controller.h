#ifndef gui_controller_h
#define gui_controller_h

class GUI;
class Engine;

class GuiController {
    public:
        static GuiController *get_instance();

        void init(Engine *engine);
        void launch();
        void update();
        void quit();

    private:
        static GuiController *instance;
        GuiController();

        GUI *gui;
};

#endif
