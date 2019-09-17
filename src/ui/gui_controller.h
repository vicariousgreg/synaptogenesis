#ifndef gui_controller_h
#define gui_controller_h

class GuiController {
    public:
        static void launch();
        static void update();
        static void quit(bool signal=true);
};

#ifndef __GUI__

void GuiController::launch() { }
void GuiController::update() { }
void GuiController::quit(bool signal) { }

#endif

#endif
