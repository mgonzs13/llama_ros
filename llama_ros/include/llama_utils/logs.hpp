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

#ifndef LLAMA_UTILS__LOGS_HPP
#define LLAMA_UTILS__LOGS_HPP

#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace llama_utils {

/**
 * @brief Type definition for a logging function.
 *
 * This type represents a function pointer that takes a file name,
 * function name, line number, log message, and a variable number of
 * additional arguments for formatting the log message.
 *
 * @param file The name of the source file where the log function is called.
 * @param function The name of the function where the log function is called.
 * @param line The line number in the source file where the log function is
 * called.
 * @param text The format string for the log message, similar to printf.
 * @param ... Additional arguments for the format string.
 */
typedef void (*LogFunction)(const char *file, const char *function, int line,
                            const char *text, ...);

// Declare function pointers for logging at different severity levels
extern LogFunction log_error; ///< Pointer to the error logging function
extern LogFunction log_warn;  ///< Pointer to the warning logging function
extern LogFunction log_info;  ///< Pointer to the info logging function
extern LogFunction log_debug; ///< Pointer to the debug logging function

/**
 * @brief Enum representing different log levels for controlling log verbosity.
 *
 * This enum defines the severity levels of logs that can be used to control
 * which log messages should be displayed. The levels are ordered from most
 * severe to least severe. Only logs at or above the current log level will be
 * shown.
 */
enum LogLevel {
  /// Log level for error messages. Only critical errors should be logged.
  ERROR = 0,
  /// Log level for warning messages. Indicate potential issues that are not
  /// critical.
  WARN,
  /// Log level for informational messages. General runtime information about
  /// the system's state.
  INFO,
  /// Log level for debug messages. Used for detailed information, mainly for
  /// developers.
  DEBUG
};

/**
 * @brief The current log level for the application.
 *
 * This global variable holds the current log level, which determines the
 * verbosity of the logs. Logs at or above this level will be displayed. The
 * default level is set to INFO.
 */
extern LogLevel log_level;

/**
 * @brief Extracts the filename from a given file path.
 *
 * This function takes a full path to a file and returns just the file name.
 *
 * @param path The full path to the file.
 * @return A pointer to the extracted filename.
 */
inline const char *extract_filename(const char *path) {
  const char *filename = std::strrchr(path, '/');
  if (!filename) {
    filename = std::strrchr(path, '\\'); // handle Windows-style paths
  }
  return filename ? filename + 1 : path;
}

#define LLAMA_LOG_ERROR(text, ...)                                             \
  if (llama_utils::log_level >= llama_utils::ERROR)                            \
  llama_utils::log_error(llama_utils::extract_filename(__FILE__),              \
                         __FUNCTION__, __LINE__, text, ##__VA_ARGS__)

#define LLAMA_LOG_WARN(text, ...)                                              \
  if (llama_utils::log_level >= llama_utils::WARN)                             \
  llama_utils::log_warn(llama_utils::extract_filename(__FILE__), __FUNCTION__, \
                        __LINE__, text, ##__VA_ARGS__)

#define LLAMA_LOG_INFO(text, ...)                                              \
  if (llama_utils::log_level >= llama_utils::INFO)                             \
  llama_utils::log_info(llama_utils::extract_filename(__FILE__), __FUNCTION__, \
                        __LINE__, text, ##__VA_ARGS__)

#define LLAMA_LOG_DEBUG(text, ...)                                             \
  if (llama_utils::log_level >= llama_utils::DEBUG)                            \
  llama_utils::log_debug(llama_utils::extract_filename(__FILE__),              \
                         __FUNCTION__, __LINE__, text, ##__VA_ARGS__)

/**
 * @brief Sets the log level for the logs.
 *
 * This function allows the user to specify the log level error, warning, info,
 * or debug.
 *
 * @param log_level Log level.
 */
void set_log_level(LogLevel log_level);

} // namespace llama_utils

#endif // llama_utils__LOGS_HPP