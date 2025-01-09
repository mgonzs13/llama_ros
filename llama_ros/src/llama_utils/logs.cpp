// Copyright (C) 2024 Miguel Ángel González Santamarta
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "llama_utils/logs.hpp"

namespace llama_utils {

/**
 * @brief Default error logging function.
 *
 * This function logs an error message to stderr with the format:
 * [ERROR] [file:function:line] message.
 *
 * @param file The name of the source file where the log function is called.
 * @param function The name of the function where the log function is called.
 * @param line The line number in the source file where the log function is
 * called.
 * @param text The format string for the log message.
 * @param ... Additional arguments for the format string.
 */
void default_log_error(const char *file, const char *function, int line,
                       const char *text, ...) {
  va_list args;
  va_start(args, text);
  fprintf(stderr, "[ERROR] [%s:%s:%d] ", file, function, line);
  vfprintf(stderr, text, args);
  fprintf(stderr, "\n");
  va_end(args);
}

/**
 * @brief Default warning logging function.
 *
 * This function logs a warning message to stderr with the format:
 * [WARN] [file:function:line] message.
 *
 * @param file The name of the source file where the log function is called.
 * @param function The name of the function where the log function is called.
 * @param line The line number in the source file where the log function is
 * called.
 * @param text The format string for the log message.
 * @param ... Additional arguments for the format string.
 */
void default_log_warn(const char *file, const char *function, int line,
                      const char *text, ...) {
  va_list args;
  va_start(args, text);
  fprintf(stderr, "[WARN] [%s:%s:%d] ", file, function, line);
  vfprintf(stderr, text, args);
  fprintf(stderr, "\n");
  va_end(args);
}

/**
 * @brief Default info logging function.
 *
 * This function logs an informational message to stderr with the format:
 * [INFO] [file:function:line] message.
 *
 * @param file The name of the source file where the log function is called.
 * @param function The name of the function where the log function is called.
 * @param line The line number in the source file where the log function is
 * called.
 * @param text The format string for the log message.
 * @param ... Additional arguments for the format string.
 */
void default_log_info(const char *file, const char *function, int line,
                      const char *text, ...) {
  va_list args;
  va_start(args, text);
  fprintf(stderr, "[INFO] [%s:%s:%d] ", file, function, line);
  vfprintf(stderr, text, args);
  fprintf(stderr, "\n");
  va_end(args);
}

/**
 * @brief Default debug logging function.
 *
 * This function logs a debug message to stderr with the format:
 * [DEBUG] [file:function:line] message.
 *
 * @param file The name of the source file where the log function is called.
 * @param function The name of the function where the log function is called.
 * @param line The line number in the source file where the log function is
 * called.
 * @param text The format string for the log message.
 * @param ... Additional arguments for the format string.
 */
void default_log_debug(const char *file, const char *function, int line,
                       const char *text, ...) {
  va_list args;
  va_start(args, text);
  fprintf(stderr, "[DEBUG] [%s:%s:%d] ", file, function, line);
  vfprintf(stderr, text, args);
  fprintf(stderr, "\n");
  va_end(args);
}

// Initialize the function pointers with default log functions
LogFunction log_error = default_log_error;
LogFunction log_warn = default_log_warn;
LogFunction log_info = default_log_info;
LogFunction log_debug = default_log_debug;

// Initialize the log level to INFO
LogLevel log_level = INFO;

void set_log_level(LogLevel log_level) { log_level = log_level; }

} // namespace llama_utils