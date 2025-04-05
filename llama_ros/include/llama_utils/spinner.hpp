// MIT License
//
// Copyright (c) 2023 Miguel Ángel González Santamarta
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef LLAMA_UTILS__SPINNER_HPP
#define LLAMA_UTILS__SPINNER_HPP

#include <cstdio>
#include <string>

namespace llama_utils {

/**
 * @class Spinner
 * @brief A utility class for displaying a spinning animation in the terminal.
 */
class Spinner {

public:
  /**
   * @brief Constructs a Spinner object.
   *
   * Initializes the spinner animation characters and sets the starting index.
   */

  Spinner() {
    this->spinner = "-\\|/";
    this->index = 0;
  }

  /**
   * @brief Displays the spinner animation with an optional text message.
   *
   * @param text The text to display alongside the spinner. Defaults to an empty
   * string.
   */
  void spin(std::string text) {
    fprintf(stderr, "%c %s\n", spinner[index], text.c_str());
    fflush(stderr);
    fprintf(stderr, "\033[1A\033[2K");
    index = (index + 1) % 4;
  }

  /**
   * @brief Displays the spinner animation without any text.
   */
  void spin() { this->spin(""); }

private:
  /**
   * @brief The spinner characters used for the animation.
   *
   * The spinner consists of four characters that are displayed in a loop to
   * create the spinning effect.
   */
  const char *spinner;

  /**
   * @brief The current index of the spinner character being displayed.
   *
   * This index is used to determine which character from the spinner string to
   * display next.
   */
  int index = 0;
};

} // namespace llama_utils

#endif
