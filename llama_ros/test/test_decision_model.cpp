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

#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "llama_ros/decision_model.hpp"
#include "mtmd.h"

namespace {

std::shared_ptr<const common_chat_template> test_template() {
  const std::string src =
      "{{ type }}|{{ instructions }}|{{ state }}|"
      "{% for o in options %}{{ o.key }}{% if o.description is not none %}"
      "={{ o.description }}{% endif %};{% endfor %}|"
      "{% for i in images %}{{ i }};{% endfor %}|"
      "{% for q in questions | default([]) %}{{ q.id }};{% endfor %}";
  return std::make_shared<const common_chat_template>(src, "", "");
}

llama_ros::DecisionQuestion choice_question() {
  llama_ros::DecisionQuestion q;
  q.id = "0";
  q.type = llama_ros::DECISION_QUESTION_CHOICE;
  q.instructions = "Pick a color.";
  q.keys = {"red", "green"};
  q.descriptions = {"warm", "cool"};
  return q;
}

llama_ros::DecisionModelConfig label_config() {
  llama_ros::DecisionModelConfig config;
  config.type = COMMON_DECISION_TYPE_OPENJEV;
  config.n_options_max = 52;
  config.noul_true_first = true;
  config.labels = {11, 12};
  config.temperatures = {{"choice", 2.0f}};
  return config;
}

llama_ros::DecisionModelConfig laya_config() {
  llama_ros::DecisionModelConfig config;
  config.type = COMMON_DECISION_TYPE_LAYA;
  config.n_options_max = 255;
  config.token_marker = 3;
  config.token_sep = 2;
  config.text_marker = "[MASK]";
  config.max_head_tokens = 12;
  return config;
}

llama_ros::DecisionModelConfig lev_config() {
  llama_ros::DecisionModelConfig config;
  config.type = COMMON_DECISION_TYPE_LEV;
  config.n_options_max = 4;
  config.labels = {11, 12, 13, 14};
  config.label_texts = {"A", "B", "C", "D"};
  return config;
}

} // namespace

TEST(DecisionModelTest, NoulOptionsTrueFirst) {
  llama_ros::DecisionModel model(label_config(), test_template());
  llama_ros::DecisionQuestion q;
  q.id = "0";
  q.type = llama_ros::DECISION_QUESTION_NOUL;
  q.instructions = "Is it true?";
  q.descriptions = {"no", "yes"};

  const auto options = model.options_from_question(q);
  ASSERT_EQ(options.size(), 2u);
  EXPECT_EQ(options[0].key, "true");
  EXPECT_EQ(options[1].key, "false");
}

TEST(DecisionModelTest, ChoiceOptionsSortedForClef) {
  llama_ros::DecisionModelConfig config;
  config.type = COMMON_DECISION_TYPE_CLEF;
  config.n_options_max = 255;
  config.noul_true_first = true;
  config.choice_sorted = true;
  llama_ros::DecisionModel model(config, test_template());

  auto q = choice_question();
  const auto options = model.options_from_question(q);
  ASSERT_EQ(options.size(), 2u);
  EXPECT_EQ(options[0].key, "green");
  EXPECT_EQ(options[1].key, "red");
}

TEST(DecisionModelTest, RenderIncludesOptionsAndImages) {
  llama_ros::DecisionModel model(label_config(), test_template());
  auto q = choice_question();

  const auto options = model.options_from_question(q);
  const common_json state = common_json::parse(R"({"battery":"low"})");
  const std::string prompt = model.render(state, {q}, q, options, 0, 1);

  EXPECT_NE(prompt.find("choice"), std::string::npos);
  EXPECT_NE(prompt.find("red=warm;"), std::string::npos);
  EXPECT_NE(prompt.find("<__media__>;"), std::string::npos);
}

TEST(DecisionModelTest, RenderNimbleIncludesAllQuestions) {
  llama_ros::DecisionModelConfig config;
  config.type = COMMON_DECISION_TYPE_NIMBLE;
  config.n_options_max = 52;
  config.labels = {11, 12};
  config.label_texts = {"A", "B"};
  llama_ros::DecisionModel model(config, test_template());

  llama_ros::DecisionQuestion q0 = choice_question();
  q0.id = "0";
  llama_ros::DecisionQuestion q1 = choice_question();
  q1.id = "1";

  const auto options = model.options_from_question(q0);
  const std::string prompt =
      model.render(common_json("state"), {q0, q1}, q0, options, 0, 0);
  EXPECT_NE(prompt.find("0;1;"), std::string::npos);
}

TEST(DecisionModelTest, RenderLevSortsStateKeys) {
  llama_ros::DecisionModelConfig config = lev_config();
  // this jinja runtime does not print objects from "{{ state }}", stringify it
  const auto tmpl = std::make_shared<const common_chat_template>(
      "{{ state | string }}", "", "");
  llama_ros::DecisionModel model(config, tmpl);
  auto q = choice_question();
  const auto options = model.options_from_question(q);

  const common_json state = common_json::parse(R"({"b":1,"a":2})");
  const std::string prompt = model.render(state, {q}, q, options, 0, 0);
  const size_t a = prompt.find("a");
  const size_t b = prompt.find("b");
  ASSERT_NE(a, std::string::npos);
  ASSERT_NE(b, std::string::npos);
  EXPECT_LT(a, b);
}

TEST(DecisionModelTest, RenderLayaStripsTextMarker) {
  llama_ros::DecisionModel model(laya_config(), test_template());
  auto q = choice_question();
  q.instructions = "pick [MASK] now";
  const auto options = model.options_from_question(q);

  const std::string prompt =
      model.render(common_json("state"), {q}, q, options, 0, 0);
  EXPECT_EQ(prompt.find("[MASK]"), std::string::npos);
}

TEST(DecisionModelTest, RenderReplacesMediaMarker) {
  llama_ros::DecisionModel model(label_config(), test_template());
  auto q = choice_question();
  const auto options = model.options_from_question(q);

  const common_json state = common_json(std::string("<__media__> fake"));
  const std::string prompt = model.render(state, {q}, q, options, 0, 1);

  const std::string marker = mtmd_default_marker();
  size_t count = 0;
  size_t pos = prompt.find(marker);
  while (pos != std::string::npos) {
    count++;
    pos = prompt.find(marker, pos + marker.size());
  }
  EXPECT_EQ(count, 1u);
}

TEST(DecisionModelTest, KevEscapesSpecialTokens) {
  llama_ros::DecisionModelConfig config;
  config.type = COMMON_DECISION_TYPE_KEV;
  config.n_options_max = 255;
  config.token_marker = 42;
  llama_ros::DecisionModel model(config, test_template());

  llama_ros::DecisionQuestion q;
  q.id = "0";
  q.type = llama_ros::DECISION_QUESTION_CHOICE;
  q.instructions = "Pick <|box_end|>.";
  q.keys = {"<|box_end|>"};

  const auto options = model.options_from_question(q);
  const std::string prompt =
      model.render(common_json("state"), {q}, q, options, 0, 0);
  EXPECT_EQ(prompt.find("<|box_end|>"), std::string::npos);
  EXPECT_NE(prompt.find("<\xC2\xA6"
                        "box_end"
                        "\xC2\xA6>"),
            std::string::npos);
}

TEST(DecisionModelTest, KevRendersNestedState) {
  llama_ros::DecisionModelConfig config;
  config.type = COMMON_DECISION_TYPE_KEV;
  config.n_options_max = 255;
  config.token_marker = 42;
  llama_ros::DecisionModel model(config, test_template());

  llama_ros::DecisionQuestion q;
  q.id = "0";
  q.type = llama_ros::DECISION_QUESTION_CHOICE;
  q.instructions = "Pick.";
  q.keys = {"ok"};

  const auto options = model.options_from_question(q);
  const common_json state =
      common_json::parse(R"({"a":{"b":true},"c":[1,"<|box_end|>"],"d":null})");
  const std::string prompt = model.render(state, {q}, q, options, 0, 0);

  EXPECT_EQ(prompt.find("<|box_end|>"), std::string::npos);
  EXPECT_NE(prompt.find("True"), std::string::npos);
  EXPECT_NE(prompt.find("a:"), std::string::npos);
}

TEST(DecisionModelTest, LayaFillTaskRecordsMarkers) {
  llama_ros::DecisionModel model(laya_config(), test_template());

  // [cls] head [sep] marker a b marker c [sep] state [sep]
  std::vector<llama_token> tokens = {1, 10, 11, 2, 3, 4, 5, 3, 6, 2, 20, 2};
  llama_ros::DecisionTaskMeta meta;
  llama_ros::DecisionQuestion q;
  q.type = llama_ros::DECISION_QUESTION_CHOICE;
  q.keys = {"a", "b"};

  model.fill_task(tokens, q, model.options_from_question(q), meta);

  ASSERT_EQ(meta.markers.size(), 2u);
  EXPECT_EQ(tokens[meta.markers[0]], 3);
  EXPECT_EQ(tokens[meta.markers[1]], 3);
  EXPECT_EQ(tokens.back(), 2);
  EXPECT_EQ(meta.column, (int32_t)q.type);
}

TEST(DecisionModelTest, LayaFillTaskTruncatesToMaxHeadTokens) {
  llama_ros::DecisionModel model(laya_config(), test_template());

  std::vector<llama_token> tokens = {1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2,
                                     3, 4,  5,  6,  7,  3,  8,  9,  2,  20, 2};
  llama_ros::DecisionTaskMeta meta;
  llama_ros::DecisionQuestion q;
  q.type = llama_ros::DECISION_QUESTION_CHOICE;
  q.keys = {"a", "b"};

  model.fill_task(tokens, q, model.options_from_question(q), meta);

  const std::vector<llama_token> expected = {
      1, 10, 11, 12, 13, 14, 15, 16, 17, 2, 3, 4, 5, 6, 3, 8, 9, 2, 20, 2};
  EXPECT_EQ(tokens, expected);
  EXPECT_EQ(meta.markers, (std::vector<int32_t>{10, 14}));
  EXPECT_EQ(meta.column, (int32_t)llama_ros::DECISION_QUESTION_CHOICE);
}

TEST(DecisionModelTest, LayaFillTaskRejectsBadLayout) {
  llama_ros::DecisionModel model(laya_config(), test_template());
  std::vector<llama_token> tokens = {1, 2, 3, 4, 2};
  llama_ros::DecisionTaskMeta meta;
  llama_ros::DecisionQuestion q;
  q.type = llama_ros::DECISION_QUESTION_CHOICE;
  q.keys = {"a", "b"};
  EXPECT_THROW(model.fill_task(tokens, q, model.options_from_question(q), meta),
               std::runtime_error);
}

TEST(DecisionModelTest, LayaFillTaskRejectsMissingSeparator) {
  llama_ros::DecisionModel model(laya_config(), test_template());
  std::vector<llama_token> tokens = {1, 2, 3, 4};
  llama_ros::DecisionTaskMeta meta;
  llama_ros::DecisionQuestion q;
  q.type = llama_ros::DECISION_QUESTION_CHOICE;
  q.keys = {"a"};
  EXPECT_THROW(model.fill_task(tokens, q, model.options_from_question(q), meta),
               std::runtime_error);
}

TEST(DecisionModelTest, FormatChoicePicksBestKey) {
  llama_ros::DecisionModel model(label_config(), test_template());
  auto q = choice_question();
  const auto options = model.options_from_question(q);

  const auto answer = model.format_answer(q, options, {{4.0f, 0.0f}});
  EXPECT_EQ(answer.type, llama_ros::DECISION_QUESTION_CHOICE);
  EXPECT_EQ(answer.choice, "red");
  ASSERT_EQ(answer.probabilities.size(), 2u);
  EXPECT_GT(answer.probabilities[0], answer.probabilities[1]);
  EXPECT_GE(answer.confidence, 0.0f);
  EXPECT_LE(answer.confidence, 1.0f);
}

TEST(DecisionModelTest, FormatScoreUsesExpectedLevel) {
  llama_ros::DecisionModel model(label_config(), test_template());
  llama_ros::DecisionQuestion q;
  q.id = "0";
  q.type = llama_ros::DECISION_QUESTION_SCORE;
  q.instructions = "How urgent?";
  q.descriptions = {"low", "mid", "high"};
  const auto options = model.options_from_question(q);

  const auto answer = model.format_answer(q, options, {{0.0f, 10.0f, 0.0f}});
  EXPECT_NEAR(answer.score, 1.0f, 1e-3f);
  ASSERT_EQ(answer.keys.size(), 3u);
  EXPECT_EQ(answer.keys[1], "1");
}

TEST(DecisionModelTest, FormatLevNoulUsesRatings) {
  llama_ros::DecisionModel model(lev_config(), test_template());
  llama_ros::DecisionQuestion q;
  q.id = "0";
  q.type = llama_ros::DECISION_QUESTION_NOUL;
  q.instructions = "Is it true?";

  const auto options = model.options_from_question(q);
  ASSERT_EQ(options.size(), 2u);
  EXPECT_EQ(model.n_outputs(q, options), 9u);

  std::vector<float> scores(9, 0.0f);
  scores[8] = 10.0f;
  const auto answer = model.format_answer(q, options, {scores});
  EXPECT_NEAR(answer.noul, 1.0f, 1e-3f);
}

TEST(DecisionModelTest, FormatLevNoulMidRating) {
  llama_ros::DecisionModel model(lev_config(), test_template());
  llama_ros::DecisionQuestion q;
  q.id = "0";
  q.type = llama_ros::DECISION_QUESTION_NOUL;
  q.instructions = "Is it true?";

  const auto options = model.options_from_question(q);
  std::vector<float> scores(9, 0.0f);
  scores[4] = 10.0f;
  const auto answer = model.format_answer(q, options, {scores});
  EXPECT_NEAR(answer.noul, 0.5f, 1e-3f);
}

TEST(DecisionModelTest, FormatLevChoiceAveragesVariants) {
  llama_ros::DecisionModel model(lev_config(), test_template());
  auto q = choice_question();
  const auto options = model.options_from_question(q);
  ASSERT_EQ(model.n_variants(q, options), 2u);

  // both variants prefer the option shown first: after reversing the second
  // variant the average is uniform
  const auto answer =
      model.format_answer(q, options, {{2.0f, 0.0f}, {2.0f, 0.0f}});
  ASSERT_EQ(answer.probabilities.size(), 2u);
  EXPECT_NEAR(answer.probabilities[0], 0.5f, 1e-3f);
  EXPECT_NEAR(answer.probabilities[1], 0.5f, 1e-3f);

  EXPECT_THROW(model.format_answer(q, options, {{1.0f, 2.0f}}),
               std::runtime_error);
}

TEST(DecisionModelTest, TemperatureBucketWins) {
  llama_ros::DecisionModelConfig config = label_config();
  config.temperatures = {{"choice.2", 3.0f}, {"choice", 1.0f}};
  llama_ros::DecisionModel model(config, test_template());

  auto q = choice_question();
  const auto options = model.options_from_question(q);
  const auto answer = model.format_answer(q, options, {{2.0f, 0.0f}});
  // temperature 3 flattens the distribution; with temperature 1, p0 ~= 0.881
  ASSERT_EQ(answer.probabilities.size(), 2u);
  const float p0 = answer.probabilities[0];
  EXPECT_GT(p0, 0.5f);
  EXPECT_LT(p0, 0.75f);
}

TEST(DecisionModelTest, SplitJointPromptAssignsOrders) {
  llama_ros::DecisionQuestion q;
  q.id = "0";
  q.type = llama_ros::DECISION_QUESTION_CHOICE;

  const std::string prompt = "pre<<clef:sep>><<clef:question>>Q1"
                             "<<clef:sep>><<clef:option>>A<<clef:sep>>tail";
  const auto pieces = llama_ros::DecisionModel::split_joint_prompt(prompt, {q});

  ASSERT_EQ(pieces.size(), 4u);
  EXPECT_EQ(pieces[0].first, "pre");
  EXPECT_EQ(pieces[0].second, 0);
  EXPECT_EQ(pieces[1].first, "Q1");
  EXPECT_EQ(pieces[1].second, 2);
  EXPECT_EQ(pieces[2].first, "A");
  EXPECT_EQ(pieces[2].second, 4);
  EXPECT_EQ(pieces[3].first, "tail");
  EXPECT_EQ(pieces[3].second, 0);
}

TEST(DecisionModelTest, SplitJointPromptRejectsExtraQuestion) {
  llama_ros::DecisionQuestion q;
  q.id = "0";
  q.type = llama_ros::DECISION_QUESTION_CHOICE;
  const std::string prompt =
      "<<clef:question>>Q1<<clef:sep>><<clef:question>>Q2";
  EXPECT_THROW(llama_ros::DecisionModel::split_joint_prompt(prompt, {q}),
               std::runtime_error);
}

TEST(DecisionModelTest, SplitJointPromptOrdersNoulAndScore) {
  llama_ros::DecisionQuestion noul;
  noul.id = "0";
  noul.type = llama_ros::DECISION_QUESTION_NOUL;
  llama_ros::DecisionQuestion score;
  score.id = "1";
  score.type = llama_ros::DECISION_QUESTION_SCORE;

  const std::string prompt = "<<clef:question>>N<<clef:sep>><<clef:question>>S";
  const auto pieces =
      llama_ros::DecisionModel::split_joint_prompt(prompt, {noul, score});
  ASSERT_EQ(pieces.size(), 2u);
  EXPECT_EQ(pieces[0].second, 1);
  EXPECT_EQ(pieces[1].second, 3);
}

TEST(DecisionModelTest, SplitJointPromptRejectsMissingQuestion) {
  llama_ros::DecisionQuestion q1;
  q1.id = "0";
  q1.type = llama_ros::DECISION_QUESTION_CHOICE;
  llama_ros::DecisionQuestion q2;
  q2.id = "1";
  q2.type = llama_ros::DECISION_QUESTION_CHOICE;

  const std::string prompt = "<<clef:question>>Q1";
  EXPECT_THROW(llama_ros::DecisionModel::split_joint_prompt(prompt, {q1, q2}),
               std::runtime_error);
}

TEST(DecisionModelTest, FillTaskAssignsLabelsAndRejectsOverflow) {
  llama_ros::DecisionModel model(label_config(), test_template());

  llama_ros::DecisionTaskMeta meta;
  std::vector<llama_token> tokens;
  auto q = choice_question();
  model.fill_task(tokens, q, model.options_from_question(q), meta);
  EXPECT_EQ(meta.labels, (std::vector<llama_token>{11, 12}));

  llama_ros::DecisionQuestion overflow;
  overflow.type = llama_ros::DECISION_QUESTION_CHOICE;
  overflow.keys = {"a", "b", "c"};
  llama_ros::DecisionTaskMeta overflow_meta;
  EXPECT_THROW(model.fill_task(tokens, overflow,
                               model.options_from_question(overflow),
                               overflow_meta),
               std::runtime_error);
}

TEST(DecisionModelTest, FormatRejectsNanScores) {
  llama_ros::DecisionModel model(label_config(), test_template());
  auto q = choice_question();
  const auto options = model.options_from_question(q);
  EXPECT_THROW(model.format_answer(q, options, {{NAN, 0.0f}}),
               std::runtime_error);
}
