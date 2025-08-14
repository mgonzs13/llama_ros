// MIT License
//
// Copyright (c) 2024 Miguel Ángel González Santamarta
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

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef LLAVA_ROS__LLAVA_HPP
#define LLAVA_ROS__LLAVA_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"
#include "mtmd-helper.h"
#include "mtmd.h"

#include "llama_ros/llama.hpp"

namespace llava_ros {

/**
 * @brief Represents the Llava model, extending the Llama model with image
 * processing capabilities.
 *
 * This class provides additional functionality for handling images and
 * generating embeddings.
 */
class Llava : public llama_ros::Llama {

public:
  /**
   * @brief Constructs a new Llava instance.
   *
   * @param params Common parameters for the llama.cpp.
   * @param system_prompt The system prompt to initialize the model's context.
   */
  Llava(const struct common_params &params, std::string system_prompt = "");

  /**
   * @brief Destroys the Llava instance.
   *
   * Cleans up resources associated with the Llava model.
   */
  ~Llava();

  /**
   * @brief Resets the internal state of the Llava model.
   *
   * This method overrides the reset functionality of the base Llama class.
   */
  void reset() override;

  /**
   * @brief Loads an mtmd into the Llava model.
   *
   * @param std::vector<uint8_t> buf The mtmd data as a byte buffer.
   * @return True if the mtmd is successfully loaded, false otherwise.
   */
  bool load_mtmd(std::vector<uint8_t> buf);

  /**
   * @brief Loads an mtmd into the Llava model.
   *
   * @param std::vector<std::vector<uint8_t>> mtmd The mtmds data as a vector
   * of
   * @return True if the image is successfully loaded, false otherwise.
   */
  bool load_mtmds(std::vector<std::vector<uint8_t>> mtmds);

  /**
   * @brief Clears all loaded mtmds from the Llava model.
   */
  void clear_mtmds();

protected:
  /**
   * @brief Loads a prompt into the Llava model.
   *
   * This method overrides the base Llama class to load a prompt into the Llava
   * model, with optional prefix and suffix handling.
   *
   * @param input_prompt The input text prompt to load.
   * @param add_pfx Whether to add a prefix to the prompt.
   * @param add_sfx Whether to add a suffix to the prompt.
   */
  void load_prompt(const std::string &input_prompt, bool add_pfx,
                   bool add_sfx) override;

  /**
   * @brief Evaluates a specific mtmd chunk in the Llava model.
   *
   * This method processes the provided mtmd chunk and integrates it into the
   * model's context.
   *
   * @param image_chunk The mtmd chunk to evaluate.
   * @return True if the mtmd chunk evaluation is successful, false
   * otherwise.
   */
  bool eval_mtmd_chunk(const mtmd_input_chunk *image_chunk);

  /**
   * @brief Evaluates the input prompt in the Llava model.
   *
   * This method overrides the base Llama class to evaluate the input prompt,
   * including image-related context.
   *
   * @return True if the prompt evaluation is successful, false otherwise.
   */
  bool eval_prompt(llama_ros::ServerSlot *slot) override;

  /**
   * @brief Pointer to the multimodal context used for image processing.
   *
   * This context is used for managing the state and operations of the
   * multimodal.
   */
  struct mtmd_context *mtmd_ctx;

private:
  /**
   * @brief Bitmaps for image processing.
   *
   * This structure holds the bitmap data for images used in the model.
   */
  mtmd::bitmaps bitmaps;

  mtmd::input_chunks chunks;
};

} // namespace llava_ros

#endif
