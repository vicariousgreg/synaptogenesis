#include "util/error_manager.h"

#include <iostream>

ErrorManager *ErrorManager::instance = 0;

void ErrorManager::log_warning(std::string error) {
    if (warnings)
        std::cout << "WARNING: \n" << error << "\n";
}

void ErrorManager::log_error(std::string error) {
    std::cout << "FATAL ERROR: \n" << error << "\n";
    std::cout << "Exiting...\n";
    std::terminate();
}

ErrorManager* ErrorManager::get_instance() {
    if (ErrorManager::instance == nullptr)
        ErrorManager::instance = new ErrorManager();
    return ErrorManager::instance;
}
