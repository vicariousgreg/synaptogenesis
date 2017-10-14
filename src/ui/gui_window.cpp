#include "gui_window.h"
#include "gui.h"
#include "engine/engine.h"

GuiObject::GuiObject() { GUI::get_instance()->init(); }
GuiWindow::GuiWindow() { GUI::get_instance()->add_window(this); }

bool GuiWindow::on_delete_event(GdkEventAny* any_event) {
    GUI::get_instance()->hide_windows();
    Engine::interrupt();
}
