#include "util/error_manager.h"

#include <iostream>
#include <stdexcept>

ErrorManager *ErrorManager::instance = 0;

void ErrorManager::log_warning(std::string error) {
    if (warnings and not suppress_output)
        std::cout << "WARNING: \n" << error << "\n";
}

void ErrorManager::log_error(std::string error) {
    if (not suppress_output)
        std::cout << "\n============\nFATAL ERROR: \n" << error << "\n";

    if (debug) throw DebugError();
    else       throw std::runtime_error(error);
}

void ErrorManager::log_debug(std::string error) {
    if (debug and not suppress_output)
        std::cout << "DEBUG: \n" << error << "\n";
}

ErrorManager* ErrorManager::get_instance() {
    if (ErrorManager::instance == nullptr)
        ErrorManager::instance = new ErrorManager();
    return ErrorManager::instance;
}
