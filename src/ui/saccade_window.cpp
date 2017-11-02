#include "saccade_window.h"
#include "impl/saccade_window_impl.h"

SaccadeWindow* SaccadeWindow::build(SaccadeModule *module) {
    return new SaccadeWindowImpl(module);
}
