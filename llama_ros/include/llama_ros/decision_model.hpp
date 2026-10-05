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
 */
struct DecisionModelConfig {
  common_decision_type type = COMMON_DECISION_TYPE_NONE;
  bool supports_images = false;
  size_t n_options_max = 0;
  bool noul_true_first = false;
  bool choice_sorted = false;
  std::vector<llama_token> labels;
  std::vector<std::string> label_texts;
  llama_token token_marker = LLAMA_TOKEN_NULL;
  llama_token token_sep = LLAMA_TOKEN_NULL;
  std::string text_marker;
  size_t max_head_tokens = 0;
  std::map<std::string, float> temperatures;
};

/**
 * @brief Decision model logic, a port of upstream server-decision.cpp.
 */
class DecisionModel {
public:
  DecisionModel() = default;

  DecisionModel(const DecisionModelConfig &config,
                std::shared_ptr<const common_chat_template> tmpl)
      : config_(config), tmpl_(std::move(tmpl)) {}

  /// @brief Reads the decision metadata of a loaded model; throws on error.
  void init_from_model(const llama_model *model);

  bool enabled() const {
    return this->config_.type != COMMON_DECISION_TYPE_NONE;
  }
  common_decision_type type() const { return this->config_.type; }
  bool supports_images() const { return this->config_.supports_images; }
  bool is_joint() const {
    return this->config_.type == COMMON_DECISION_TYPE_CLEF;
  }
  bool needs_embeddings() const;
  size_t n_options_max() const { return this->config_.n_options_max; }
  llama_token get_token_marker() const { return this->config_.token_marker; }

  std::vector<DecisionOption>
  options_from_question(const DecisionQuestion &question) const;

  size_t n_variants(const DecisionQuestion &question,
                    const std::vector<DecisionOption> &options) const;

  size_t n_outputs(const DecisionQuestion &question,
                   const std::vector<DecisionOption> &options) const;

  std::string render(const common_json &state,
                     const std::vector<DecisionQuestion> &questions,
                     const DecisionQuestion &question,
                     const std::vector<DecisionOption> &options, size_t variant,
                     size_t n_images) const;

  std::string
  render_joint(const common_json &state,
               const std::vector<DecisionQuestion> &questions) const;

  void fill_task(std::vector<llama_token> &tokens,
                 const DecisionQuestion &question,
                 const std::vector<DecisionOption> &options,
                 DecisionTaskMeta &meta) const;

  void fill_task_joint(const llama_vocab *vocab,
                       const std::vector<DecisionQuestion> &questions,
                       const std::string &prompt,
                       std::vector<llama_token> &tokens,
                       DecisionTaskMeta &meta) const;

  static std::vector<std::pair<std::string, int32_t>>
  split_joint_prompt(const std::string &prompt,
                     const std::vector<DecisionQuestion> &questions);

  DecisionAnswer
  format_answer(const DecisionQuestion &question,
                const std::vector<DecisionOption> &options,
                const std::vector<std::vector<float>> &scores) const;

private:
  common_json render_options(const std::vector<DecisionOption> &options,
                             size_t variant) const;

  void fill_task_laya(std::vector<llama_token> &tokens,
                      const DecisionQuestion &question, size_t n_options,
                      DecisionTaskMeta &meta) const;

  float get_temperature(const DecisionQuestion &question,
                        const std::vector<DecisionOption> &options) const;

  DecisionModelConfig config_;
  std::shared_ptr<const common_chat_template> tmpl_;
};

} // namespace llama_ros

#endif // LLAMA_ROS__DECISION_MODEL_HPP
