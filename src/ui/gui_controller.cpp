#include "gui_controller.h"
#include "gui.h"

void GuiController::launch() {
    GUI::get_instance()->launch();
}

void GuiController::update() {
    GUI::get_instance()->signal_update();
}

void GuiController::quit() {
    GUI::get_instance()->signal_quit();
}
