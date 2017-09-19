#include "dsst_window.h"
#include "impl/dsst_window_impl.h"

DSSTWindow* DSSTWindow::build() {
    return new DSSTWindowImpl();
}
