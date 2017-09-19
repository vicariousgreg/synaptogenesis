#include "gui_controller.h"
#include "gui.h"

GuiController *GuiController::instance = new GuiController();

GuiController::GuiController() {
    this->gui = GUI::get_instance();
}

GuiController *GuiController::get_instance() {
    if (GuiController::instance == nullptr)
        GuiController::instance = new GuiController();
    return GuiController::instance;
}

void GuiController::launch() {
    gui->launch();
}

void GuiController::update() {
    gui->signal_update();
}

void GuiController::quit() {
    gui->get_instance()->quit();
}
