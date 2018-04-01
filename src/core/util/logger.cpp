#include "util/logger.h"

#include <iostream>
#include <stdexcept>

bool Logger::suppress_output = false;
bool Logger::warnings = true;
bool Logger::debug = false;

void Logger::log_warning(std::string error) {
    if (warnings and not suppress_output)
        std::cout << "WARNING: \n" << error << "\n";
}

void Logger::log_error(std::string error) {
    if (not suppress_output)
        std::cout << "\n============\nFATAL ERROR: \n" << error << "\n";

    if (debug) throw DebugError();
    else       throw std::runtime_error(error);
}

void Logger::log_debug(std::string error) {
    if (debug and not suppress_output)
        std::cout << "DEBUG: \n" << error << "\n";
}
