#include "dsst_window.h"
#include "impl/dsst_window_impl.h"

DSSTWindow* DSSTWindow::build(DSSTModule *module) {
    return new DSSTWindowImpl(module);
}
