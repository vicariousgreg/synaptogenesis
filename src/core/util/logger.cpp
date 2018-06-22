#include "util/logger.h"

#include <iostream>
#include <stdexcept>

bool Logger::suppress_output = false;
bool Logger::warnings = true;
bool Logger::debug = false;

void Logger::log_warning(std::string error, const char* file, int line) {
    if (warnings and not suppress_output)
        std::cout << "WARNING ("
            << file << ":" << line << "): \n" << error << "\n";
}

void Logger::log_error(std::string error, const char* file, int line) {
    if (not suppress_output)
        std::cout << "FATAL ERROR ("
            << file << ":" << line << "): \n" << error << "\n";

    if (debug) throw DebugError();
    else       throw std::runtime_error(error);
}

void Logger::log_debug(std::string error, const char* file, int line) {
    if (debug and not suppress_output)
        std::cout << "DEBUG ("
            << file << ":" << line << "): \n" << error << "\n";
}
