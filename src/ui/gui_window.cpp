#include "gui_window.h"
#include "gui.h"

GuiWindow::GuiWindow() { GUI::get_instance()->add_window(this); }

bool GuiWindow::on_delete_event(GdkEventAny* any_event) {
    GUI::get_instance()->interrupt_engine();
}
