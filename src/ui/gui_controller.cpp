#include "gui_controller.h"
#include "gui.h"

void GuiController::launch() {
    GUI::get_instance()->launch();
}

void GuiController::update() {
    GUI::get_instance()->signal_update();
}

void GuiController::quit(bool signal) {
    if (signal)
        GUI::get_instance()->signal_quit();
    else
        GUI::get_instance()->quit();
}
