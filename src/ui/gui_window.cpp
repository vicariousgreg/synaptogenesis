#include "gui_window.h"
#include "gui.h"

GuiWindow::GuiWindow() { GUI::get_instance()->add_window(this); }
