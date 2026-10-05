// MIT License
//
// Copyright (c) 2026 Miguel Ángel González Santamarta
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

#ifndef LLAMA_ROS__DECISION_MODEL_HPP
#define LLAMA_ROS__DECISION_MODEL_HPP

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"

#include "llama_ros/types.hpp"

namespace llama_ros {

/**
 * @brief Plain configuration of a decision model, independent of llama.cpp.
 *
 * It holds everything needed to render prompts, read model outputs and format
 * answers for one decision type. `init_from_model` fills it from the GGUF
 * metadata; tests build it by hand.
 */
struct DecisionModelConfig {
  /// @brief Upstream decision type (COMMON_DECISION_TYPE_NONE if disabled).
  common_decision_type type = COMMON_DECISION_TYPE_NONE;

  /// @brief Whether the model accepts images (OpenJEV).
  bool supports_images = false;

  /// @brief Maximum number of options accepted (label count for OpenJEV,
  /// Lev and Nimble; 255 for the other types).
  size_t n_options_max = 0;

  /// @brief Whether Noul options are ordered true, false (OpenJEV, Clef).
  bool noul_true_first = false;

  /// @brief Whether choice options are sorted by key (Clef).
  bool choice_sorted = false;

  /// @brief Single-token labels scored through the logits output
  /// (OpenJEV, Lev, Nimble).
  std::vector<llama_token> labels;

  /// @brief Text of each label given to the template (Lev, Nimble).
  std::vector<std::string> label_texts;

  /// @brief Token marking each option in the prompt (Laya mask token,
  /// Kev <|box_end|> token).
  llama_token token_marker = LLAMA_TOKEN_NULL;

  /// @brief Separator token of the Laya prompt layout.
  llama_token token_sep = LLAMA_TOKEN_NULL;

  /// @brief Text form of the marker token, stripped from user input (Laya).
  std::string text_marker;

  /// @brief Maximum number of tokens for the question and its options
  /// (Laya).
  size_t max_head_tokens = 0;

  /// @brief Softmax temperatures by "<type>" or "<type>.<n_options bucket>".
  std::map<std::string, float> temperatures;
};

/**
 * @brief Decision model logic, a port of upstream server-decision.cpp.
 *
 * A decision model answers typed questions about a state by scoring the
 * options in a single forward pass, without generating tokens. This class
 * owns the type-specific behavior:
 *
 * - OpenJEV, Lev and Nimble score single-token labels through the logits.
 * - Kev scores options through a scaled dot product of embedding rows.
 * - Laya reads one embedding column per option.
 * - Clef answers all questions of a request in one joint prompt.
 *
 * Prompt rendering uses the model's "systemone" chat template. The class is
 * independent of the ROS layer and of the slot machinery; `Llama` calls it
 * and reads the outputs according to the `DecisionTaskMeta` it fills.
 */
class DecisionModel {
public:
  /// @brief Creates a disabled model (type COMMON_DECISION_TYPE_NONE).
  DecisionModel() = default;

  /**
   * @brief Creates a model from an explicit configuration.
   *
   * Used by tests and by callers that build the configuration by hand.
   *
   * @param config The model configuration.
   * @param tmpl The "systemone" chat template. It may be null only if no
   * render method is called.
   */
  DecisionModel(const DecisionModelConfig &config,
                std::shared_ptr<const common_chat_template> tmpl)
      : config_(config), tmpl_(std::move(tmpl)) {}

  /**
   * @brief Reads the decision metadata of a loaded model.
   *
   * Reads `common_get_decision_type`, the "systemone" chat template and the
   * `<arch>.decision.*` keys, and fills the configuration. A model without
   * decision metadata leaves the instance disabled.
   *
   * @param model The loaded llama.cpp model.
   * @throws std::runtime_error On an unknown decision type, a missing
   * "systemone" template, missing metadata, or an invalid temperature.
   */
  void init_from_model(const llama_model *model);

  /**
   * @brief Checks whether a supported decision type is configured.
   *
   * @return True if the model can evaluate decisions.
   */
  bool enabled() const {
    return this->config_.type != COMMON_DECISION_TYPE_NONE;
  }

  /**
   * @brief Gets the upstream decision type.
   *
   * @return The decision type (COMMON_DECISION_TYPE_NONE if disabled).
   */
  common_decision_type type() const { return this->config_.type; }

  /**
   * @brief Checks whether the model accepts images.
   *
   * @return True if images are supported (OpenJEV).
   */
  bool supports_images() const { return this->config_.supports_images; }

  /**
   * @brief Checks whether all questions of a request share one prompt.
   *
   * @return True for Clef, false otherwise.
   */
  bool is_joint() const {
    return this->config_.type == COMMON_DECISION_TYPE_CLEF;
  }

  /**
   * @brief Checks whether the readout needs the embeddings output.
   *
   * @return True for Laya, Kev and Clef, false for label-logit types.
   */
  bool needs_embeddings() const;

  /**
   * @brief Gets the maximum number of options accepted.
   *
   * @return The maximum option count.
   */
  size_t n_options_max() const { return this->config_.n_options_max; }

  /**
   * @brief Gets the option marker token.
   *
   * @return The marker token, or LLAMA_TOKEN_NULL when the type does not use
   * one.
   */
  llama_token get_token_marker() const { return this->config_.token_marker; }

  /**
   * @brief Builds the ordered options of a question.
   *
   * Applies the type rules: choice options may be sorted by key (Clef) and
   * Noul options may be swapped to true, false (OpenJEV, Clef).
   *
   * @param question The question, already validated.
   * @return The options in the order the model sees them.
   */
  std::vector<DecisionOption>
  options_from_question(const DecisionQuestion &question) const;

  /**
   * @brief Gets the number of prompt variants used to answer a question.
   *
   * Lev shows the options of a choice question in 2 orders to cancel the
   * preference for the first label.
   *
   * @param question The question.
   * @param options The ordered options.
   * @return 2 for Lev choice questions with more than one option, else 1.
   */
  size_t n_variants(const DecisionQuestion &question,
                    const std::vector<DecisionOption> &options) const;

  /**
   * @brief Gets the number of model outputs per variant.
   *
   * @param question The question.
   * @param options The ordered options.
   * @return 9 for Lev Noul questions (rating scale), else the option count.
   */
  size_t n_outputs(const DecisionQuestion &question,
                   const std::vector<DecisionOption> &options) const;

  /**
   * @brief Renders the "systemone" prompt of one question variant.
   *
   * @param state The state shared by all questions (JSON or plain text).
   * @param questions All questions of the request (used by Nimble).
   * @param question The question to render.
   * @param options The ordered options.
   * @param variant The variant index (0, or 1 for reversed Lev options).
   * @param n_images Number of image markers to insert in the prompt.
   * @return The rendered prompt.
   * @throws std::runtime_error When the model has no template.
   */
  std::string render(const common_json &state,
                     const std::vector<DecisionQuestion> &questions,
                     const DecisionQuestion &question,
                     const std::vector<DecisionOption> &options, size_t variant,
                     size_t n_images) const;

  /**
   * @brief Renders the joint prompt of all questions (Clef).
   *
   * @param state The state shared by all questions.
   * @param questions All questions of the request.
   * @return The rendered joint prompt with the Clef markers.
   * @throws std::runtime_error When the model has no template.
   */
  std::string
  render_joint(const common_json &state,
               const std::vector<DecisionQuestion> &questions) const;

  /**
   * @brief Fills the token prompt and readout metadata of one question.
   *
   * For Laya it truncates the prompt to `max_head_tokens` and records the
   * option marker positions and column; for Kev it records the option marker
   * positions and the pointer row; for label types it selects the label
   * tokens to read from the logits.
   *
   * @param tokens The tokenized prompt, modified in place.
   * @param question The question.
   * @param options The ordered options.
   * @param meta The readout metadata, modified in place.
   * @throws std::runtime_error On an unexpected prompt layout or when there
   * are more options than labels.
   */
  void fill_task(std::vector<llama_token> &tokens,
                 const DecisionQuestion &question,
                 const std::vector<DecisionOption> &options,
                 DecisionTaskMeta &meta) const;

  /**
   * @brief Tokenizes a joint prompt and fills its readout metadata (Clef).
   *
   * The prompt is tokenized piece by piece so each token gets a decision
   * order, and the number of scored options is recorded.
   *
   * @param vocab The model vocabulary.
   * @param questions All questions of the request, in order.
   * @param prompt The joint prompt from `render_joint`.
   * @param tokens The output token sequence.
   * @param meta The readout metadata, modified in place.
   * @throws std::invalid_argument When a question or option span is empty.
   * @throws std::runtime_error On an unexpected prompt layout.
   */
  void fill_task_joint(const llama_vocab *vocab,
                       const std::vector<DecisionQuestion> &questions,
                       const std::string &prompt,
                       std::vector<llama_token> &tokens,
                       DecisionTaskMeta &meta) const;

  /**
   * @brief Splits a joint prompt into pieces with their decision order.
   *
   * The order values match the upstream LLAMA_DECISION_ORDER_* enum:
   * 0 = not scored, 1 = Noul question, 2 = choice question, 3 = score
   * question, 4 = option.
   *
   * @param prompt The joint prompt from `render_joint`.
   * @param questions All questions of the request, in order.
   * @return One (piece, order) pair per template piece.
   * @throws std::runtime_error When the prompt does not cover all questions
   * or has more question markers than questions.
   */
  static std::vector<std::pair<std::string, int32_t>>
  split_joint_prompt(const std::string &prompt,
                     const std::vector<DecisionQuestion> &questions);

  /**
   * @brief Computes the answer of a question from the raw model outputs.
   *
   * Applies a temperature-scaled softmax per variant, averages the variants
   * (reversing the second one), and derives the choice, score or Noul value
   * plus the confidence.
   *
   * @param question The question.
   * @param options The ordered options.
   * @param scores One raw score vector per variant.
   * @return The formatted answer.
   * @throws std::runtime_error On a variant/option count mismatch or when
   * the model returned NaN scores.
   */
  DecisionAnswer
  format_answer(const DecisionQuestion &question,
                const std::vector<DecisionOption> &options,
                const std::vector<std::vector<float>> &scores) const;

private:
  /**
   * @brief Builds the JSON options array given to the template.
   *
   * @param options The ordered options.
   * @param variant The variant index (1 reverses the option order).
   * @return The options as a JSON array, with Kev escaping and label texts.
   */
  common_json render_options(const std::vector<DecisionOption> &options,
                             size_t variant) const;

  /**
   * @brief Truncates and rearranges a Laya prompt to `max_head_tokens`.
   *
   * @param tokens The tokenized prompt, modified in place.
   * @param question The question (sets the readout column).
   * @param n_options The number of options in the prompt.
   * @param meta The readout metadata, modified in place.
   * @throws std::runtime_error On an unexpected prompt layout.
   */
  void fill_task_laya(std::vector<llama_token> &tokens,
                      const DecisionQuestion &question, size_t n_options,
                      DecisionTaskMeta &meta) const;

  /**
   * @brief Gets the temperature to apply to a question.
   *
   * Looks up "<type>.<bucket>" first and falls back to "<type>", then 1.0.
   *
   * @param question The question.
   * @param options The ordered options (the option count selects the bucket).
   * @return The temperature.
   */
  float get_temperature(const DecisionQuestion &question,
                        const std::vector<DecisionOption> &options) const;

  /// @brief The model configuration.
  DecisionModelConfig config_;

  /// @brief The "systemone" chat template (null when not initialized).
  std::shared_ptr<const common_chat_template> tmpl_;
};

} // namespace llama_ros

#endif // LLAMA_ROS__DECISION_MODEL_HPP
