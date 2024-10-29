// MIT License

// Copyright (c) 2024  Miguel Ángel González Santamarta

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fstream>

#include "common.h"
#include "json-schema-to-grammar.h"
#include "json.hpp"

#include "llama_utils/llama_params.hpp"

using namespace llama_utils;

void replace_all(std::string &input, const std::string &old_str,
                 const std::string &new_str) {
  size_t start_pos = 0;
  while ((start_pos = input.find(old_str, start_pos)) != std::string::npos) {
    input.replace(start_pos, old_str.length(), new_str);
    start_pos += new_str.length();
  }
}

void llama_utils::declare_llama_params(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr &node) {

  node->declare_parameters<int32_t>("", {
                                            {"seed", -1},
                                            {"n_ctx", 512},
                                            {"n_batch", 2048},
                                            {"n_ubatch", 512},
                                            {"n_gpu_layers", 0},
                                            {"main_gpu", 0},
                                            {"n_threads", 1},
                                            {"poll", 50},
                                            {"n_threads_batch", 1},
                                            {"poll_batch", 50},
                                            {"n_predict", 128},
                                            {"n_keep", -1},
                                            {"grp_attn_n", 1},
                                            {"grp_attn_w", 512},
                                            {"n_parallel", 1},
                                            {"n_sequences", 1},
                                            {"yarn_orig_ctx", 0},
                                        });
  node->declare_parameters<std::string>("", {
                                                {"model", ""},
                                                {"mmproj", ""},
                                                {"cpu_mask", ""},
                                                {"cpu_range", ""},
                                                {"cpu_mask_batch", ""},
                                                {"cpu_range_batch", ""},
                                                {"priority", "normal"},
                                                {"priority_batch", "normal"},
                                                {"split_mode", "layer"},
                                                {"rope_scaling_type", ""},
                                                {"numa", "none"},
                                                {"pooling_type", ""},
                                                {"cache_type_k", "f16"},
                                                {"cache_type_v", "f16"},
                                                {"system_prompt", ""},
                                                {"system_prompt_file", ""},
                                                {"prefix", ""},
                                                {"suffix", ""},
                                                {"image_prefix", ""},
                                                {"image_suffix", ""},
                                                {"image_text", "<image>"},
                                            });
  node->declare_parameter<std::vector<std::string>>(
      "lora_adapters", std::vector<std::string>({}));
  node->declare_parameter<std::vector<double>>("lora_adapters_scales",
                                               std::vector<double>({}));
  node->declare_parameter<std::vector<std::string>>(
      "stopping_words", std::vector<std::string>({}));
  node->declare_parameters<float>("", {
                                          {"rope_freq_base", 0.0f},
                                          {"rope_freq_scale", 0.0f},
                                          {"yarn_ext_factor", -1.0f},
                                          {"yarn_attn_factor", 1.0f},
                                          {"yarn_beta_fast", 32.0f},
                                          {"yarn_beta_slow", 1.0f},
                                      });
  node->declare_parameter<std::vector<double>>("tensor_split",
                                               std::vector<double>({0.0}));
  node->declare_parameters<bool>("", {
                                         {"debug", true},
                                         {"embedding", false},
                                         {"reranking", false},
                                         {"logits_all", false},
                                         {"use_mmap", true},
                                         {"use_mlock", false},
                                         {"cont_batching", true},
                                         {"no_kv_offload", false},
                                         {"warmup", true},
                                         {"check_tensors", false},
                                         {"flash_attn", false},
                                         {"strict_cpu", false},
                                         {"strict_cpu_batch", false},
                                     });
}

struct llama_params llama_utils::get_llama_params(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr &node) {

  int32_t seed;
  int32_t poll;
  int32_t poll_batch;

  std::vector<std::string> lora_adapters;
  std::vector<double> lora_adapters_scales;
  std::vector<std::string> stopping_words;
  std::vector<double> tensor_split;

  std::string cpu_mask;
  std::string cpu_range;
  std::string cpu_mask_batch;
  std::string cpu_range_batch;

  std::string priority;
  std::string priority_batch;
  std::string split_mode;
  std::string rope_scaling_type;
  std::string numa;
  std::string pooling_type;

  std::string file_path;

  struct llama_params params;

  node->get_parameter("seed", seed);
  node->get_parameter("n_ctx", params.params.n_ctx);
  node->get_parameter("n_batch", params.params.n_batch);
  node->get_parameter("n_ubatch", params.params.n_ubatch);

  node->get_parameter("n_gpu_layers", params.params.n_gpu_layers);
  node->get_parameter("split_mode", split_mode);
  node->get_parameter("main_gpu", params.params.main_gpu);
  node->get_parameter("tensor_split", tensor_split);

  node->get_parameter("embedding", params.params.embedding);
  node->get_parameter("reranking", params.params.reranking);
  node->get_parameter("logits_all", params.params.logits_all);
  node->get_parameter("use_mmap", params.params.use_mmap);
  node->get_parameter("use_mlock", params.params.use_mlock);
  node->get_parameter("warmup", params.params.warmup);
  node->get_parameter("check_tensors", params.params.check_tensors);
  node->get_parameter("flash_attn", params.params.flash_attn);

  node->get_parameter("no_kv_offload", params.params.no_kv_offload);
  node->get_parameter("cache_type_k", params.params.cache_type_k);
  node->get_parameter("cache_type_v", params.params.cache_type_v);

  node->get_parameter("n_threads", params.params.cpuparams.n_threads);
  node->get_parameter("cpu_mask", cpu_mask);
  node->get_parameter("cpu_range", cpu_range);
  node->get_parameter("priority", priority);
  node->get_parameter("strict_cpu", params.params.cpuparams.strict_cpu);
  node->get_parameter("poll", poll);

  node->get_parameter("n_threads_batch",
                      params.params.cpuparams_batch.n_threads);
  node->get_parameter("cpu_mask_batch", cpu_mask_batch);
  node->get_parameter("cpu_range_batch", cpu_range_batch);
  node->get_parameter("priority_batch", priority_batch);
  node->get_parameter("strict_cpu_batch",
                      params.params.cpuparams_batch.strict_cpu);
  node->get_parameter("poll_batch", poll_batch);

  node->get_parameter("n_predict", params.params.n_predict);
  node->get_parameter("n_keep", params.params.n_keep);
  node->get_parameter("n_batch", params.params.n_batch);

  node->get_parameter("grp_attn_n", params.params.grp_attn_n);
  node->get_parameter("grp_attn_w", params.params.grp_attn_w);

  node->get_parameter("rope_freq_base", params.params.rope_freq_base);
  node->get_parameter("rope_freq_scale", params.params.rope_freq_scale);
  node->get_parameter("rope_scaling_type", rope_scaling_type);

  node->get_parameter("yarn_ext_factor", params.params.yarn_ext_factor);
  node->get_parameter("yarn_attn_factor", params.params.yarn_attn_factor);
  node->get_parameter("yarn_beta_fast", params.params.yarn_beta_fast);
  node->get_parameter("yarn_beta_slow", params.params.yarn_beta_slow);
  node->get_parameter("yarn_orig_ctx", params.params.yarn_orig_ctx);

  node->get_parameter("model", params.params.model);
  node->get_parameter("lora_adapters", lora_adapters);
  node->get_parameter("lora_adapters_scales", lora_adapters_scales);
  node->get_parameter("mmproj", params.params.mmproj);
  node->get_parameter("numa", numa);
  node->get_parameter("pooling_type", pooling_type);

  node->get_parameter("n_parallel", params.params.n_parallel);
  node->get_parameter("n_sequences", params.params.n_sequences);
  node->get_parameter("cont_batching", params.params.cont_batching);

  node->get_parameter("prefix", params.params.input_prefix);
  node->get_parameter("suffix", params.params.input_suffix);
  node->get_parameter("stopping_words", stopping_words);
  node->get_parameter("image_prefix", params.llava_params.image_prefix);
  node->get_parameter("image_suffix", params.llava_params.image_suffix);
  node->get_parameter("image_text", params.llava_params.image_text);

  node->get_parameter("system_prompt", params.system_prompt);
  node->get_parameter("system_prompt_file", file_path);
  node->get_parameter("debug", params.debug);

  // seed
  if (seed < 0) {
    params.params.sparams.seed = LLAMA_DEFAULT_SEED;
  } else {
    params.params.sparams.seed = seed;
  }

  // check threads number
  if (params.params.cpuparams.n_threads < 0) {
    params.params.cpuparams.n_threads = cpu_get_num_math();
  }

  if (params.params.cpuparams_batch.n_threads < 0) {
    params.params.cpuparams_batch.n_threads = cpu_get_num_math();
  }

  // lora_adapters
  if (lora_adapters.size()) {
    if (lora_adapters.size() != lora_adapters_scales.size()) {
      RCLCPP_ERROR(
          node->get_logger(),
          "lora_adapters and lora_adapters_scales must have the same size");
    } else {

      for (size_t i = 0; i < lora_adapters.size(); i++) {

        if (lora_adapters.at(i).empty()) {
          continue;
        }

        float scale = (float)lora_adapters_scales.at(i);

        if (scale < 0.0) {
          RCLCPP_WARN(node->get_logger(),
                      "Scale %f cannot be lower than 0.0, setting it to 0.0",
                      scale);
          scale = 0.0;
        } else if (scale > 1.0) {
          RCLCPP_WARN(node->get_logger(),
                      "Scale %f cannot be greater than 1.0, setting it to 1.0",
                      scale);
          scale = 1.0;
        }

        params.params.lora_adapters.push_back({lora_adapters.at(i), scale});
      }
    }
  }

  // stopping words are the antiprompt
  for (std::string word : stopping_words) {

    if (word.empty()) {
      continue;
    }

    replace_all(word, "\\n", "\n");
    params.params.antiprompt.push_back(word);
  }

  // split mode
  if (split_mode == "none") {
    params.params.split_mode = LLAMA_SPLIT_MODE_NONE;
  } else if (split_mode == "layer") {
    params.params.split_mode = LLAMA_SPLIT_MODE_LAYER;
  } else if (split_mode == "row") {
    params.params.split_mode = LLAMA_SPLIT_MODE_ROW;
  }

  // cpu mask
  if (!cpu_mask.empty()) {
    params.params.cpuparams.mask_valid = true;
    parse_cpu_mask(cpu_mask, params.params.cpuparams.cpumask);
  }

  if (!cpu_range.empty()) {
    params.params.cpuparams.mask_valid = true;
    parse_cpu_mask(cpu_range, params.params.cpuparams.cpumask);
  }

  if (!cpu_mask_batch.empty()) {
    params.params.cpuparams_batch.mask_valid = true;
    parse_cpu_mask(cpu_mask_batch, params.params.cpuparams_batch.cpumask);
  }

  if (!cpu_range_batch.empty()) {
    params.params.cpuparams_batch.mask_valid = true;
    parse_cpu_mask(cpu_range_batch, params.params.cpuparams_batch.cpumask);
  }

  // cpu priority
  params.params.cpuparams.priority = parse_priority(priority);
  params.params.cpuparams_batch.priority = parse_priority(priority_batch);

  // cpu poll
  params.params.cpuparams.poll = poll;
  params.params.cpuparams_batch.poll = poll_batch;

  // rerank
  if (params.params.reranking) {
    params.params.embedding = true;
  }

  // rope_scaling_type
  if (rope_scaling_type == "none") {
    params.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
  } else if (rope_scaling_type == "linear") {
    params.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
  } else if (rope_scaling_type == "yarn") {
    params.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
  } else {
    params.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
  }

  // numa
  if (numa == "none") {
    params.params.numa = GGML_NUMA_STRATEGY_DISABLED;
  } else if (numa == "distribute") {
    params.params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
  } else if (numa == "isolate") {
    params.params.numa = GGML_NUMA_STRATEGY_ISOLATE;
  } else if (numa == "numactl") {
    params.params.numa = GGML_NUMA_STRATEGY_NUMACTL;
  } else if (numa == "mirror") {
    params.params.numa = GGML_NUMA_STRATEGY_MIRROR;
  } else if (numa == "count") {
    params.params.numa = GGML_NUMA_STRATEGY_COUNT;
  }

  // pooling
  if (pooling_type == "none") {
    params.params.pooling_type = LLAMA_POOLING_TYPE_NONE;
  } else if (pooling_type == "mean") {
    params.params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  } else if (pooling_type == "cls") {
    params.params.pooling_type = LLAMA_POOLING_TYPE_CLS;
  } else if (pooling_type == "last") {
    params.params.pooling_type = LLAMA_POOLING_TYPE_LAST;
  } else {
    params.params.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
  }

  // initial prompt
  if (!file_path.empty()) {
    std::ifstream file(file_path.c_str());
    if (!file) {
      RCLCPP_ERROR(node->get_logger(), "Failed to open file %s",
                   file_path.c_str());
    }
    std::copy(std::istreambuf_iterator<char>(file),
              std::istreambuf_iterator<char>(),
              back_inserter(params.system_prompt));
  }

  // split tensors
  GGML_ASSERT(tensor_split.size() <= llama_max_devices());
  for (size_t i = 0; i < llama_max_devices(); ++i) {
    if (i < tensor_split.size()) {
      params.params.tensor_split[i] = tensor_split[i];
    } else {
      params.params.tensor_split[i] = 0.0f;
    }
  }

  return params;
}

enum ggml_sched_priority llama_utils::parse_priority(std::string priority) {
  if (priority == "normal") {
    return GGML_SCHED_PRIO_NORMAL;
  } else if (priority == "medium") {
    return GGML_SCHED_PRIO_MEDIUM;
  } else if (priority == "high") {
    return GGML_SCHED_PRIO_HIGH;
  } else if (priority == "realtime") {
    return GGML_SCHED_PRIO_REALTIME;
  }

  return GGML_SCHED_PRIO_NORMAL;
}

struct common_sampler_params llama_utils::parse_sampling_params(
    const llama_msgs::msg::SamplingConfig &sampling_config, int n_vocab) {

  struct common_sampler_params sparams;

  sparams.n_prev = sampling_config.n_prev;
  sparams.n_probs = sampling_config.n_probs;

  sparams.temp = sampling_config.temp;

  sparams.top_k = sampling_config.top_k;
  sparams.top_p = sampling_config.top_p;
  sparams.min_p = sampling_config.min_p;
  sparams.typ_p = sampling_config.typical_p;

  sparams.penalty_last_n = sampling_config.penalty_last_n;
  sparams.penalty_repeat = sampling_config.penalty_repeat;
  sparams.penalty_freq = sampling_config.penalty_freq;
  sparams.penalty_present = sampling_config.penalty_present;

  sparams.mirostat = sampling_config.mirostat;
  sparams.mirostat_eta = sampling_config.mirostat_eta;
  sparams.mirostat_tau = sampling_config.mirostat_tau;

  sparams.penalize_nl = sampling_config.penalize_nl;

  sparams.samplers =
      common_sampler_types_from_chars(sampling_config.samplers_sequence);
  sparams.grammar = sampling_config.grammar;

  if (sparams.grammar.size() == 0 &&
      sampling_config.grammar_schema.size() > 0) {

    sparams.grammar = json_schema_to_grammar(
        nlohmann::ordered_json::parse(sampling_config.grammar_schema));
  }

  // check penalty_last_n
  sparams.penalty_last_n =
      sparams.penalty_last_n < 0 ? sparams.n_prev : sparams.penalty_last_n;

  // check top_k
  sparams.top_k = sparams.top_k <= 0 ? n_vocab : sparams.top_k;

  // add logit bias
  for (auto logit_bias : sampling_config.logit_bias.data) {
    sparams.logit_bias.push_back({logit_bias.token, logit_bias.bias});
  }

  return sparams;
}
