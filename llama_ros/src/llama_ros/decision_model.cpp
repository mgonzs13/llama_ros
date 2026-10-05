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

#include "llama_ros/decision_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <regex>
#include <stdexcept>

#include "llama_utils/logs.hpp"
#include "mtmd.h"

using namespace llama_ros;

namespace {

constexpr size_t DECISION_LEV_N_RATINGS = 9;

const std::string CLEF_MARKER = "<<clef:";
const std::string CLEF_SEP = "<<clef:sep>>";
const std::string CLEF_MARK_QUESTION = "<<clef:question>>";
const std::string CLEF_MARK_OPTION = "<<clef:option>>";

const char *decision_question_type_name(DecisionQuestionType type) {
  switch (type) {
  case DECISION_QUESTION_CHOICE:
    return "choice";
  case DECISION_QUESTION_SCORE:
    return "score";
  case DECISION_QUESTION_NOUL:
    return "noul";
  }
  return "";
}

std::string decision_meta_str(const llama_model *model,
                              const std::string &key) {
  char buf[256];
  const int32_t n =
      llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf));
  return n < 0 ? "" : std::string(buf);
}

// replace text in all strings of a JSON value
common_json decision_replace_text(const common_json &val,
                                  const std::string &search,
                                  const std::string &replace) {
  if (val.is_string()) {
    std::string str = val.get<std::string>();
    string_replace_all(str, search, replace);
    return str;
  } else if (val.is_array()) {
    common_json out = common_json::array();
    for (const auto &item : val) {
      out.push_back(decision_replace_text(item, search, replace));
    }
    return out;
  } else if (val.is_object()) {
    common_json out = common_json::object();
    for (const auto &[key, item] : val.items()) {
      out[key] = decision_replace_text(item, search, replace);
    }
    return out;
  }

  return val;
}

// sort the keys of all objects of a JSON value
common_json decision_sort_keys(const common_json &val) {
  if (val.is_array()) {
    common_json out = common_json::array();
    for (const auto &item : val) {
      out.push_back(decision_sort_keys(item));
    }
    return out;
  }

  if (val.is_object()) {

    std::map<std::string, common_json> sorted;
    for (const auto &[key, item] : val.items()) {
      sorted[key] = decision_sort_keys(item);
    }

    common_json out = common_json::object();
    for (const auto &[key, item] : sorted) {
      out[key] = item;
    }

    return out;
  }

  return val;
}

// kev flattens a JSON value into text, the keys of an object are kept as labels
std::string decision_kev_render(const common_json &val, int indent = 0) {
  const std::string pad(2 * indent, ' ');
  if (val.is_null()) {
    return "";

  } else if (val.is_string()) {
    return val.get<std::string>();

  } else if (val.is_boolean()) {
    return val.get<bool>() ? "True" : "False";

  } else if (val.is_array()) {
    std::string out;
    for (const auto &item : val) {
      const std::string text = decision_kev_render(item, indent + 1);
      out +=
          (out.empty() ? "" : "\n") + pad + "- " +
          text.substr(std::min(text.size(), text.find_first_not_of(" \t\n\r")));
    }
    return out;

  } else if (val.is_object()) {
    std::string out;
    for (const auto &[key, item] : val.items()) {
      const bool is_nested = item.is_object() || item.is_array();
      out += (out.empty() ? "" : "\n") + pad + key +
             (is_nested ? ":\n" : ": ") +
             decision_kev_render(item, is_nested ? indent + 1 : 0);
    }
    return out;
  }

  return val.dump();
}

// kev text input: special tokens written in the text must not be parsed as such
std::string decision_kev_text(const common_json &val) {
  static const std::regex re_special("<\\|([A-Za-z0-9_]+)\\|>");
  return std::regex_replace(decision_kev_render(val), re_special,
                            "<\xC2\xA6$1\xC2\xA6>");
}

} // namespace

void DecisionModel::init_from_model(const llama_model *model) {
  this->config_ = DecisionModelConfig();
  this->tmpl_.reset();

  const common_decision_type model_type = common_get_decision_type(model);
  if (model_type == COMMON_DECISION_TYPE_NONE) {
    return;
  }

  const std::string prefix =
      decision_meta_str(model, "general.architecture") + ".decision.";
  const std::string type_name = decision_meta_str(model, prefix + "type");

  const char *tmpl_src = llama_model_chat_template(model, "systemone");

  if (tmpl_src == nullptr) {
    throw std::runtime_error("decision model has no \"systemone\" template");
  }

  this->tmpl_ = std::make_shared<const common_chat_template>(tmpl_src, "", "");

  const std::string prefix_temp = prefix + "temperature.";

  for (int32_t i = 0; i < llama_model_meta_count(model); i++) {

    char key[256];
    char val[64];

    if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 ||
        !string_starts_with(key, prefix_temp)) {
      continue;
    }

    if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0) {
      continue;
    }

    const float temp = std::strtof(val, nullptr);

    if (temp <= 0.0f) {
      throw std::runtime_error(
          string_format("invalid decision temperature: %s = %s", key, val));
    }

    this->config_.temperatures[key + prefix_temp.size()] = temp;
  }

  const llama_vocab *vocab = llama_model_get_vocab(model);

  if (model_type == COMMON_DECISION_TYPE_OPENJEV) {
    // one letter per option, each must be a single token
    const std::string letters =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    for (const char c : letters) {
      const auto toks = common_tokenize(vocab, std::string(1, c), false, false);
      if (toks.size() != 1) {
        throw std::runtime_error(
            string_format("decision label '%c' is not a single token", c));
      }
      this->config_.labels.push_back(toks[0]);
    }

    this->config_.n_options_max = this->config_.labels.size();
    this->config_.noul_true_first = true;
    this->config_.supports_images = true;

  } else if (model_type == COMMON_DECISION_TYPE_LEV ||
             model_type == COMMON_DECISION_TYPE_NIMBLE) {
    // label codes are A..Z then AA..ZZ, only single-token ones are used
    std::vector<std::string> codes;
    for (char a = 'A'; a <= 'Z'; a++) {
      codes.push_back(std::string(1, a));
    }

    for (char a = 'A'; a <= 'Z'; a++) {
      for (char b = 'A'; b <= 'Z'; b++) {
        codes.push_back(std::string{a, b});
      }
    }

    for (const auto &code : codes) {
      const auto toks = common_tokenize(vocab, code, false, false);
      if (toks.size() == 1 && this->config_.labels.size() < 255) {
        this->config_.labels.push_back(toks[0]);
        this->config_.label_texts.push_back(code);
      }
    }

    this->config_.n_options_max = this->config_.labels.size();

  } else if (model_type == COMMON_DECISION_TYPE_KEV) {
    // the hidden state of an option is read at the token that ends it
    const auto toks = common_tokenize(vocab, "<|box_end|>", false, true);
    if (toks.size() != 1) {
      throw std::runtime_error("decision model has no <|box_end|> token");
    }
    this->config_.token_marker = toks[0];
    this->config_.n_options_max = 255;

  } else if (model_type == COMMON_DECISION_TYPE_LAYA) {
    this->config_.token_marker = llama_vocab_mask(vocab);
    this->config_.token_sep = llama_vocab_sep(vocab);

    if (this->config_.token_marker == LLAMA_TOKEN_NULL ||
        this->config_.token_sep == LLAMA_TOKEN_NULL) {
      throw std::runtime_error("decision model has no mask or sep token");
    }

    this->config_.text_marker =
        common_token_to_piece(vocab, this->config_.token_marker, true);

    const std::string val =
        decision_meta_str(model, prefix + "max_head_tokens");
    this->config_.max_head_tokens = std::strtoul(val.c_str(), nullptr, 10);

    if (this->config_.max_head_tokens == 0) {
      throw std::runtime_error("decision model has no valid max_head_tokens");
    }
    this->config_.n_options_max = 255;

  } else if (model_type == COMMON_DECISION_TYPE_CLEF) {
    this->config_.n_options_max = 255;
    this->config_.noul_true_first = true;
    this->config_.choice_sorted = true;
    this->config_.supports_images = true;

  } else {
    throw std::runtime_error("unsupported decision model type: " + type_name);
  }

  this->config_.type = model_type;
  LLAMA_LOG_INFO("Decision model type: %s", type_name.c_str());
}

bool DecisionModel::needs_embeddings() const {
  switch (this->config_.type) {
  case COMMON_DECISION_TYPE_LAYA:
  case COMMON_DECISION_TYPE_KEV:
  case COMMON_DECISION_TYPE_CLEF:
    return true;
  default:
    return false;
  }
}

std::vector<DecisionOption>
DecisionModel::options_from_question(const DecisionQuestion &question) const {
  std::vector<DecisionOption> options;

  switch (question.type) {
  case DECISION_QUESTION_CHOICE: {
    for (size_t i = 0; i < question.keys.size(); ++i) {
      options.push_back({question.keys[i], i < question.descriptions.size()
                                               ? question.descriptions[i]
                                               : ""});
    }
    if (this->config_.choice_sorted) {
      std::sort(options.begin(), options.end(),
                [](const DecisionOption &a, const DecisionOption &b) {
                  return a.key < b.key;
                });
    }
    break;
  }

  case DECISION_QUESTION_SCORE: {
    for (size_t i = 0; i < question.descriptions.size(); ++i) {
      options.push_back({std::to_string(i), question.descriptions[i]});
    }
    break;
  }

  case DECISION_QUESTION_NOUL: {
    options.push_back({"false", question.descriptions.size() > 0
                                    ? question.descriptions[0]
                                    : ""});
    options.push_back({"true", question.descriptions.size() > 1
                                   ? question.descriptions[1]
                                   : ""});
    if (this->config_.noul_true_first) {
      std::swap(options[0], options[1]);
    }
    break;
  }
  }

  return options;
}

size_t
DecisionModel::n_variants(const DecisionQuestion &question,
                          const std::vector<DecisionOption> &options) const {
  // lev shows the options of a choice in 2 orders, to cancel the preference
  // for the first label
  if (this->config_.type == COMMON_DECISION_TYPE_LEV &&
      question.type == DECISION_QUESTION_CHOICE && options.size() > 1) {
    return 2;
  }
  return 1;
}

size_t
DecisionModel::n_outputs(const DecisionQuestion &question,
                         const std::vector<DecisionOption> &options) const {
  if (this->config_.type == COMMON_DECISION_TYPE_LEV &&
      question.type == DECISION_QUESTION_NOUL) {
    return DECISION_LEV_N_RATINGS;
  }
  return options.size();
}

common_json
DecisionModel::render_options(const std::vector<DecisionOption> &options,
                              size_t variant) const {
  const size_t n_options = options.size();

  common_json out = common_json::array();

  for (size_t i = 0; i < n_options; i++) {
    // the second variant shows the options in the reverse order
    const auto &opt = options[variant == 0 ? i : n_options - 1 - i];
    common_json option = common_json::object();
    option["key"] = opt.key;
    option["description"] =
        opt.description.empty() ? common_json() : common_json(opt.description);

    if (this->config_.type == COMMON_DECISION_TYPE_KEV) {
      option["key"] = decision_kev_text(opt.key);
      if (!opt.description.empty()) {
        option["description"] = decision_kev_text(opt.description);
      }
    }

    if (i < this->config_.label_texts.size()) {
      option["label"] = this->config_.label_texts[i];
    }

    out.push_back(option);
  }

  return out;
}

std::string
DecisionModel::render(const common_json &state,
                      const std::vector<DecisionQuestion> &questions,
                      const DecisionQuestion &question,
                      const std::vector<DecisionOption> &options,
                      size_t variant, size_t n_images) const {

  common_json inp = common_json::object();
  inp["id"] = question.id;
  inp["type"] = std::string(decision_question_type_name(question.type));
  inp["instructions"] = question.instructions;
  inp["state"] = state;
  inp["options"] = this->render_options(options, variant);

  // the nimble prompt lists all the questions of the request
  if (this->config_.type == COMMON_DECISION_TYPE_NIMBLE) {
    common_json qs = common_json::array();

    for (const auto &q : questions) {
      common_json qj = common_json::object();
      qj["id"] = q.id;
      qj["type"] = std::string(decision_question_type_name(q.type));
      qj["instructions"] = q.instructions;
      qj["options"] = this->render_options(this->options_from_question(q), 0);
      qs.push_back(qj);
    }

    inp["questions"] = qs;
  }

  // lev was trained with sorted keys
  if (this->config_.type == COMMON_DECISION_TYPE_LEV) {
    inp = decision_sort_keys(inp);
  }

  // the kev template only takes text
  if (this->config_.type == COMMON_DECISION_TYPE_KEV) {
    inp["state"] = decision_kev_text(state);
    inp["instructions"] = decision_kev_text(question.instructions);
  }

  // the input must not contain the marker of the options
  if (!this->config_.text_marker.empty()) {
    inp = decision_replace_text(inp, this->config_.text_marker, " ");
  }

  // the template puts one media marker per image
  common_json images = common_json::array();
  if (n_images > 0) {
    inp = decision_replace_text(inp, mtmd_default_marker(), " ");
    for (size_t i = 0; i < n_images; i++) {
      images.push_back(mtmd_default_marker());
    }
  }
  inp["images"] = images;

  if (this->tmpl_ == nullptr) {
    throw std::runtime_error("decision model has no template");
  }
  jinja::context ctx(this->tmpl_->source());
  jinja::global_from_json(ctx, inp, false);
  jinja::runtime runtime(ctx);
  const jinja::value results = runtime.execute(this->tmpl_->prog);
  return jinja::runtime::gather_string_parts(results)->as_string().str();
}

void DecisionModel::fill_task(std::vector<llama_token> &tokens,
                              const DecisionQuestion &question,
                              const std::vector<DecisionOption> &options,
                              DecisionTaskMeta &meta) const {
  meta = DecisionTaskMeta{};

  switch (this->config_.type) {
  case COMMON_DECISION_TYPE_OPENJEV:
  case COMMON_DECISION_TYPE_LEV:
  case COMMON_DECISION_TYPE_NIMBLE: {
    const size_t n = this->n_outputs(question, options);
    if (n > this->config_.labels.size()) {
      throw std::runtime_error("too many options for this decision model");
    }
    meta.labels.assign(this->config_.labels.begin(),
                       this->config_.labels.begin() + n);
    return;
  }
  case COMMON_DECISION_TYPE_KEV: {
    // an option is read at its end token, the question at the last token
    for (size_t i = 0; i < tokens.size(); i++) {
      if (tokens[i] == this->config_.token_marker) {
        meta.markers.push_back((int32_t)i);
      }
    }
    if (meta.markers.size() != options.size()) {
      throw std::runtime_error("unexpected layout of the decision prompt");
    }
    meta.pointer = (int32_t)tokens.size() - 1;
    return;
  }
  case COMMON_DECISION_TYPE_LAYA:
    this->fill_task_laya(tokens, question, options.size(), meta);
    return;
  default:
    throw std::runtime_error("unsupported decision model type");
  }
}

// the prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]
// options and question are cut to fit max_head_tokens, the same way the model
// was trained
void DecisionModel::fill_task_laya(std::vector<llama_token> &tokens,
                                   const DecisionQuestion &question,
                                   size_t n_options,
                                   DecisionTaskMeta &meta) const {
  constexpr size_t max_option_tokens = 48;

  meta.column = (int32_t)question.type;

  std::vector<size_t> markers;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] == this->config_.token_marker) {
      markers.push_back(i);
    }
  }

  if (n_options == 0 || markers.size() != n_options || markers[0] < 2 ||
      tokens[markers[0] - 1] != this->config_.token_sep ||
      tokens.back() != this->config_.token_sep) {
    throw std::runtime_error("unexpected layout of the decision prompt");
  }

  const size_t head_end = markers[0] - 1;
  const size_t opts_end = std::find(tokens.begin() + markers.back(),
                                    tokens.end(), this->config_.token_sep) -
                          tokens.begin();
  if (opts_end + 1 >= tokens.size()) {
    throw std::runtime_error("unexpected layout of the decision prompt");
  }

  std::vector<std::vector<llama_token>> options;
  size_t n_options_tokens = 0;
  auto set_max = [&](size_t n_max) {
    n_options_tokens = 0;
    for (auto &option : options) {
      option.resize(std::min(option.size(), n_max));
      n_options_tokens += option.size();
    }
  };

  for (size_t i = 0; i < n_options; ++i) {
    const size_t end = i + 1 < n_options ? markers[i + 1] : opts_end;
    options.emplace_back(tokens.begin() + markers[i], tokens.begin() + end);
  }

  set_max(max_option_tokens + 1);
  if (n_options_tokens + 16 > this->config_.max_head_tokens) {
    // too many or too long options, shrink them evenly
    set_max(std::max((size_t)4,
                     (this->config_.max_head_tokens -
                      std::min(this->config_.max_head_tokens, (size_t)16)) /
                         n_options));
  }

  const size_t n_question_max = std::max(
      (size_t)8, this->config_.max_head_tokens -
                     std::min(this->config_.max_head_tokens, n_options_tokens));

  std::vector<llama_token> out;
  out.push_back(tokens[0]);
  out.insert(out.end(), tokens.begin() + 1,
             tokens.begin() + std::min(head_end, 1 + n_question_max));
  out.push_back(this->config_.token_sep);

  for (const auto &option : options) {
    meta.markers.push_back((int32_t)out.size());
    out.insert(out.end(), option.begin(), option.end());
  }

  out.insert(out.end(), tokens.begin() + opts_end, tokens.end());
  tokens = std::move(out);
}

std::string
DecisionModel::render_joint(const common_json &state,
                            const std::vector<DecisionQuestion> &questions,
                            size_t n_images) const {

  if (this->tmpl_ == nullptr) {
    throw std::runtime_error("decision model has no template");
  }

  common_json inp_questions = common_json::array();
  for (const auto &question : questions) {
    common_json options = common_json::array();
    for (const auto &opt : this->options_from_question(question)) {
      common_json option = common_json::object();
      option["key"] = opt.key;
      option["description"] = opt.description.empty()
                                  ? common_json()
                                  : common_json(opt.description);
      options.push_back(option);
    }
    common_json q = common_json::object();
    q["id"] = question.id;
    q["type"] = std::string(decision_question_type_name(question.type));
    q["instructions"] = question.instructions;
    q["options"] = options;
    inp_questions.push_back(q);
  }

  // the template is given raw JSON values with sorted keys, and no marker
  common_json inp = common_json::object();
  inp["state"] = state;
  inp["questions"] = inp_questions;
  inp = decision_replace_text(decision_sort_keys(inp), CLEF_MARKER, "<<clef ");

  // the template puts one media marker per image
  common_json images = common_json::array();
  if (n_images > 0) {
    inp = decision_replace_text(inp, mtmd_default_marker(), " ");
    for (size_t i = 0; i < n_images; i++) {
      images.push_back(mtmd_default_marker());
    }
  }
  inp["images"] = images;
  inp["sep"] = CLEF_SEP;
  inp["mark_question"] = CLEF_MARK_QUESTION;
  inp["mark_option"] = CLEF_MARK_OPTION;

  jinja::context ctx(this->tmpl_->source());
  jinja::global_from_json(ctx, inp, false);
  jinja::runtime runtime(ctx);
  const jinja::value results = runtime.execute(this->tmpl_->prog);
  return jinja::runtime::gather_string_parts(results)->as_string().str();
}

size_t DecisionModel::joint_head_end(
    const std::vector<std::pair<std::string, int32_t>> &pieces,
    const std::string &media_marker) {

  if (media_marker.empty()) {
    return 0;
  }

  size_t i = 0;
  while (i < pieces.size() &&
         pieces[i].first.find(media_marker) == std::string::npos) {
    i++;
  }

  if (i == pieces.size()) {
    return 0;
  }

  // the head must not contain scored pieces
  for (size_t j = 0; j <= i; j++) {
    if (pieces[j].second != 0) {
      throw std::runtime_error("unexpected layout of the decision prompt");
    }
  }

  // only one contiguous head is supported
  for (size_t j = i + 1; j < pieces.size(); j++) {
    if (pieces[j].first.find(media_marker) != std::string::npos) {
      throw std::runtime_error("unexpected layout of the decision prompt");
    }
  }

  return i + 1;
}

std::vector<std::pair<std::string, int32_t>> DecisionModel::split_joint_prompt(
    const std::string &prompt, const std::vector<DecisionQuestion> &questions) {
  std::vector<std::pair<std::string, int32_t>> out;
  size_t i_question = 0;

  // decision orders match the vendored LLAMA_DECISION_ORDER_* enum: 1 noul
  // question, 2 choice question, 3 score question, 4 option
  for (std::string piece : string_split(prompt, CLEF_SEP)) {

    int32_t order = 0;

    if (string_starts_with(piece, CLEF_MARK_QUESTION)) {
      piece = piece.substr(CLEF_MARK_QUESTION.size());

      if (i_question >= questions.size()) {
        throw std::runtime_error("unexpected layout of the decision prompt");
      }

      switch (questions[i_question++].type) {
      case DECISION_QUESTION_NOUL:
        order = 1;
        break;
      case DECISION_QUESTION_CHOICE:
        order = 2;
        break;
      case DECISION_QUESTION_SCORE:
        order = 3;
        break;
      }

    } else if (string_starts_with(piece, CLEF_MARK_OPTION)) {
      piece = piece.substr(CLEF_MARK_OPTION.size());
      order = 4;
    }
    out.emplace_back(piece, order);
  }

  if (i_question != questions.size()) {
    throw std::runtime_error("unexpected layout of the decision prompt");
  }

  return out;
}

void DecisionModel::fill_task_joint(
    const llama_vocab *vocab, const std::vector<DecisionQuestion> &questions,
    const std::string &prompt, size_t head_end,
    std::vector<llama_token> &tokens, DecisionTaskMeta &meta) const {

  const auto pieces = split_joint_prompt(prompt, questions);
  if (head_end > pieces.size()) {
    throw std::runtime_error("unexpected layout of the decision prompt");
  }

  // the head entries (media placeholders) are not read by the head
  meta.order.assign(tokens.size(), 0);
  meta.n_scores = 0;

  // the model was trained with the pieces tokenized one by one
  for (size_t i = head_end; i < pieces.size(); i++) {
    const std::string &piece = pieces[i].first;
    const int32_t order = pieces[i].second;
    const auto piece_tokens = common_tokenize(vocab, piece, false, true);

    if (order != 0 && piece_tokens.empty()) {
      throw std::invalid_argument(
          "the instructions and the options of a question must not be empty");
    }

    tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
    meta.order.resize(tokens.size(), order);
    if (order == 4) {
      meta.n_scores++;
    }
  }

  size_t n_options = 0;
  for (const auto &question : questions) {
    n_options += this->options_from_question(question).size();
  }

  if ((size_t)meta.n_scores != n_options) {
    throw std::runtime_error("unexpected layout of the decision prompt");
  }
}

float DecisionModel::get_temperature(
    const DecisionQuestion &question,
    const std::vector<DecisionOption> &options) const {
  const size_t n = options.size();
  const std::string type_name = decision_question_type_name(question.type);

  // the temperature can depend on the number of options, the buckets are the
  // ones used to fit it
  std::string bucket;

  if (this->config_.type == COMMON_DECISION_TYPE_LEV) {
    bucket = n <= 8 ? "small" : n <= 26 ? "mid" : "large";
  } else {
    bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
  }

  for (const auto &name : {type_name + "." + bucket, type_name}) {
    const auto it = this->config_.temperatures.find(name);
    if (it != this->config_.temperatures.end()) {
      return it->second;
    }
  }
  return 1.0f;
}

// confidence formulas are the ones published by TypeSafe
DecisionAnswer DecisionModel::format_answer(
    const DecisionQuestion &question,
    const std::vector<DecisionOption> &options,
    const std::vector<std::vector<float>> &scores) const {

  const size_t n = this->n_outputs(question, options);

  if (scores.size() != this->n_variants(question, options)) {
    throw std::runtime_error(
        "decision result does not match the number of variants");
  }

  // softmax over the outputs of each variant, then the average of the variants
  const float temperature = this->get_temperature(question, options);
  std::vector<double> probs(n, 0.0);

  for (size_t v = 0; v < scores.size(); v++) {
    const auto &s = scores[v];

    if (s.size() != n) {
      throw std::runtime_error(
          "decision result does not match the number of options");
    }

    // a joint head returns NaN if it could not use the decision order
    if (std::any_of(s.begin(), s.end(),
                    [](float value) { return std::isnan(value); })) {
      throw std::runtime_error("the model could not evaluate the decision");
    }

    const float score_max = *std::max_element(s.begin(), s.end());
    std::vector<double> p(n);
    double sum = 0.0;

    for (size_t i = 0; i < n; i++) {
      p[i] = std::exp((double)(s[i] - score_max) / temperature);
      sum += p[i];
    }

    for (size_t i = 0; i < n; i++) {
      // the second variant is in the reverse order
      probs[v == 0 ? i : n - 1 - i] += p[i] / sum / scores.size();
    }
  }

  DecisionAnswer answer;
  answer.type = question.type;

  if (question.type == DECISION_QUESTION_NOUL) {
    if (this->config_.type == COMMON_DECISION_TYPE_LEV) {
      double expected = 0.0;
      for (size_t i = 0; i < n; i++) {
        expected += probs[i] * i / (n - 1);
      }
      answer.noul = (float)expected;
      return answer;
    }

    for (size_t i = 0; i < n; i++) {
      if (options[i].key == "true") {
        answer.noul = (float)probs[i];
      }
    }

    for (size_t i = 0; i < options.size(); i++) {
      answer.keys.push_back(options[i].key);
      answer.probabilities.push_back((float)probs[i]);
    }

    return answer;
  }

  for (size_t i = 0; i < n; i++) {
    answer.keys.push_back(options[i].key);
    answer.probabilities.push_back((float)probs[i]);
  }

  if (question.type == DECISION_QUESTION_CHOICE) {
    const size_t best =
        std::max_element(probs.begin(), probs.end()) - probs.begin();
    answer.choice = options[best].key;

    if (n >= 2) {
      const double uniform = 1.0 / n;
      const double p_max = *std::max_element(probs.begin(), probs.end());
      answer.confidence =
          (float)std::max(0.0, (p_max - uniform) / (1.0 - uniform));
    } else {
      answer.confidence = 1.0f;
    }

    return answer;
  }

  double expected = 0.0;
  for (size_t i = 0; i < n; ++i) {
    expected += i * probs[i];
  }
  answer.score = (float)expected;

  if (n >= 2) {
    const size_t mode =
        std::max_element(probs.begin(), probs.end()) - probs.begin();
    double dist = 0.0;
    double dist_uniform = 0.0;
    for (size_t i = 0; i < n; ++i) {
      dist += probs[i] * std::fabs((double)i - (double)mode);
      dist_uniform += std::fabs((double)i - (n - 1) / 2.0) / n;
    }
    answer.confidence = (float)std::max(0.0, 1.0 - dist / dist_uniform);

  } else {
    answer.confidence = 1.0f;
  }

  return answer;
}
