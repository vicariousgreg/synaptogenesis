#include "util/error_manager.h"

#include <iostream>
#include <stdexcept>

bool ErrorManager::suppress_output = false;
bool ErrorManager::warnings = true;
bool ErrorManager::debug = false;

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
