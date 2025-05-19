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
   * @brief Loads an image into the Llava model.
   *
   * @param base64_str The image encoded as a Base64 string.
   * @return True if the image is successfully loaded, false otherwise.
   */
  bool load_image(std::vector<uint8_t> buf);

  /**
   * @brief Converts a Base64-encoded image string into an image embedding.
   *
   * @param base64_str The image encoded as a Base64 string.
   * @return A pointer to the generated image embedding structure.
   */
  struct llava_image_embed *
  base64_image_to_embed(const std::string &base64_str);

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
   * @brief Evaluates an image embedding in the Llava model.
   *
   * This method processes the provided image embedding and integrates it into
   * the model's context.
   *
   * @return True if the image evaluation is successful, false otherwise.
   */
  bool eval_image();

  /**
   * @brief Evaluates the input prompt in the Llava model.
   *
   * This method overrides the base Llama class to evaluate the input prompt,
   * including image-related context.
   *
   * @return True if the prompt evaluation is successful, false otherwise.
   */
  bool eval_prompt() override;

  /**
   * @brief Pointer to the image embedding structure.
   *
   * This structure holds the embedding data for the currently loaded image.
   */
  struct llava_image_embed *image_embed;

  /**
   * @brief Pointer to the multimodal context used for image processing.
   *
   * This context is used for managing the state and operations of the
   * multimodal.
   */
  struct mtmd_context *mtmd_ctx;

private:
  /**
   * @brief Free the image chunk used for processing.
   *
   * This method releases the resources associated with the image chunk.
   */
  void free_image_chunk();

  /**
   * @brief Bitmaps for image processing.
   *
   * This structure holds the bitmap data for images used in the model.
   */
  mtmd::bitmaps bitmaps;

  /**
   * @brief Pointer to the image chunk used for processing.
   *
   * This structure holds the chunk of image data used in the model's context.
   */
  mtmd_input_chunk *image_chunk;

  /**
   * @brief The pose of the image in the model's context.
   *
   * This integer represents the position or state of the image in the model's
   * processing pipeline.
   */
  int image_pose;
};

} // namespace llava_ros

#endif
