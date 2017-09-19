#ifndef gui_controller_h
#define gui_controller_h

class GUI;

class GuiController {
    public:
        static GuiController *get_instance();

        void launch();
        void update();
        void quit();

    private:
        static GuiController *instance;
        GuiController();

        GUI *gui;
};

#endif
