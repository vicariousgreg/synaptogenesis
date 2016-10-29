#include <gtkmm.h>

#include "model/layer.h"
#include "ui.h"

int launch(int argc, char *argv[]) {
  auto app =
    Gtk::Application::create(argc, argv,
      "org.gtkmm.examples.base");

  Gtk::Window window;
  window.set_default_size(200, 200);

  return app->run(window);
}
