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

#include <fstream>
#include <iostream>

#include "common.h"
#include "huggingface_hub.h"
#include "json.hpp"

#include "json-schema-to-grammar.h"
#include "llama_utils/llama_params.hpp"
#include "llama_utils/logs.hpp"

using namespace llama_utils;

void replace_all(std::string &input, const std::string &old_str,
                 const std::string &new_str) {
  size_t start_pos = 0;
  while ((start_pos = input.find(old_str, start_pos)) != std::string::npos) {
    input.replace(start_pos, old_str.length(), new_str);
    start_pos += new_str.length();
  }
}

const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,    GGML_TYPE_F16,  GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,   GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const std::string &s) {
  for (const auto &type : kv_cache_types) {
    if (ggml_type_name(type) == s) {
      return type;
    }
  }
  throw std::runtime_error("Unsupported cache type: " + s);
}

std::string download_model(const std::string &repo_id,
                           const std::string &filename) {

  if (repo_id.empty() || filename.empty()) {
    return "";
  }

  auto result = huggingface_hub::hf_hub_download_with_shards(repo_id, filename);

  if (result.success) {
    return result.path;
  } else {
    return "";
  }
}

void llama_utils::declare_llama_params(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr &node) {

  node->declare_parameters<int32_t>("", {
                                            {"verbosity", 0},
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
  node->declare_parameters<std::string>(
      "", {
              {"model_path", ""},      {"model_repo", ""},
              {"model_filename", ""},  {"mmproj_path", ""},
              {"mmproj_repo", ""},     {"mmproj_filename", ""},
              {"cpu_mask", ""},        {"cpu_range", ""},
              {"cpu_mask_batch", ""},  {"cpu_range_batch", ""},
              {"priority", "normal"},  {"priority_batch", "normal"},
              {"split_mode", "layer"}, {"rope_scaling_type", ""},
              {"numa", "none"},        {"pooling_type", ""},
              {"cache_type_k", "f16"}, {"cache_type_v", "f16"},
              {"system_prompt", ""},   {"system_prompt_file", ""},
              {"prefix", ""},          {"suffix", ""},
          });
  node->declare_parameters<std::vector<std::string>>(
      {""}, {
                {"devices", std::vector<std::string>({})},
                {"stopping_words", std::vector<std::string>({})},
                {"lora_adapters", std::vector<std::string>({})},
                {"lora_adapters_repos", std::vector<std::string>({})},
                {"lora_adapters_filenames", std::vector<std::string>({})},
            });
  node->declare_parameters<float>("", {
                                          {"rope_freq_base", 0.0f},
                                          {"rope_freq_scale", 0.0f},
                                          {"yarn_ext_factor", -1.0f},
                                          {"yarn_attn_factor", 1.0f},
                                          {"yarn_beta_fast", 32.0f},
                                          {"yarn_beta_slow", 1.0f},
                                          {"defrag_thold", 0.1f},
                                      });
  node->declare_parameter<std::vector<double>>("tensor_split",
                                               std::vector<double>({0.0}));
  node->declare_parameter<std::vector<double>>("lora_adapters_scales",
                                               std::vector<double>({}));
  node->declare_parameters<bool>("", {
                                         {"embedding", false},
                                         {"reranking", false},
                                         {"use_mmap", true},
                                         {"use_mlock", false},
                                         {"cont_batching", true},
                                         {"no_op_offload", false},
                                         {"no_kv_offload", false},
                                         {"warmup", true},
                                         {"check_tensors", false},
                                         {"flash_attn", false},
                                         {"strict_cpu", false},
                                         {"strict_cpu_batch", false},
                                         {"mmproj_use_gpu", true},
                                         {"no_mmproj", false},
                                     });
}

struct LlamaParams llama_utils::get_llama_params(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr &node) {

  int32_t seed;
  int32_t poll;
  int32_t poll_batch;

  std::vector<std::string> stopping_words;

  std::vector<std::string> lora_adapters;
  std::vector<std::string> lora_adapters_repos;
  std::vector<std::string> lora_adapters_filenames;
  std::vector<double> lora_adapters_scales;

  std::vector<std::string> devices;
  std::vector<double> tensor_split;

  std::string cpu_mask;
  std::string cpu_range;
  std::string cpu_mask_batch;
  std::string cpu_range_batch;

  std::string cache_type_k;
  std::string cache_type_v;

  std::string priority;
  std::string priority_batch;
  std::string split_mode;
  std::string rope_scaling_type;
  std::string numa;
  std::string pooling_type;

  std::string file_path;

  struct LlamaParams params;

  node->get_parameter("verbosity", params.params.verbosity);
  node->get_parameter("seed", seed);
  node->get_parameter("n_ctx", params.params.n_ctx);
  node->get_parameter("n_batch", params.params.n_batch);
  node->get_parameter("n_ubatch", params.params.n_ubatch);

  node->get_parameter("devices", devices);
  node->get_parameter("n_gpu_layers", params.params.n_gpu_layers);
  node->get_parameter("split_mode", split_mode);
  node->get_parameter("main_gpu", params.params.main_gpu);
  node->get_parameter("tensor_split", tensor_split);

  bool reranking = false;
  node->get_parameter("embedding", params.params.embedding);
  node->get_parameter("reranking", reranking);
  if (reranking) {
    params.params.pooling_type = LLAMA_POOLING_TYPE_RANK;
  } else {
    params.params.pooling_type = LLAMA_POOLING_TYPE_NONE;
  }
  node->get_parameter("use_mmap", params.params.use_mmap);
  node->get_parameter("use_mlock", params.params.use_mlock);
  node->get_parameter("warmup", params.params.warmup);
  node->get_parameter("check_tensors", params.params.check_tensors);
  node->get_parameter("flash_attn", params.params.flash_attn);

  node->get_parameter("no_op_offload", params.params.no_op_offload);
  node->get_parameter("no_kv_offload", params.params.no_kv_offload);
  node->get_parameter("cache_type_k", cache_type_k);
  node->get_parameter("cache_type_v", cache_type_v);

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
  node->get_parameter("defrag_thold", params.params.defrag_thold);

  node->get_parameter("mmproj_use_gpu", params.params.mmproj_use_gpu);
  node->get_parameter("no_mmproj", params.params.no_mmproj);

  node->get_parameter("model_path", params.params.model.path);
  node->get_parameter("model_repo", params.params.model.hf_repo);
  node->get_parameter("model_filename", params.params.model.hf_file);
  node->get_parameter("mmproj_path", params.params.mmproj.path);
  node->get_parameter("mmproj_repo", params.params.mmproj.hf_repo);
  node->get_parameter("mmproj_filename", params.params.mmproj.hf_file);
  node->get_parameter("lora_adapters", lora_adapters);
  node->get_parameter("lora_adapters_repos", lora_adapters_repos);
  node->get_parameter("lora_adapters_filenames", lora_adapters_filenames);
  node->get_parameter("lora_adapters_scales", lora_adapters_scales);
  node->get_parameter("numa", numa);
  node->get_parameter("pooling_type", pooling_type);

  node->get_parameter("n_parallel", params.params.n_parallel);
  node->get_parameter("n_sequences", params.params.n_sequences);
  node->get_parameter("cont_batching", params.params.cont_batching);

  node->get_parameter("prefix", params.params.input_prefix);
  node->get_parameter("suffix", params.params.input_suffix);
  node->get_parameter("stopping_words", stopping_words);

  node->get_parameter("system_prompt", params.system_prompt);
  node->get_parameter("system_prompt_file", file_path);

  // seed
  if (seed < 0) {
    params.params.sampling.seed = LLAMA_DEFAULT_SEED;
  } else {
    params.params.sampling.seed = seed;
  }

  // cache type
  params.params.cache_type_k = kv_cache_type_from_str(cache_type_k);
  params.params.cache_type_v = kv_cache_type_from_str(cache_type_v);

  // devices
  for (const std::string &d : devices) {

    if (!d.empty()) {
      auto *dev = ggml_backend_dev_by_name(d.c_str());

      if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        LLAMA_LOG_ERROR("Invalid device: %s", d.c_str());
      } else {
        params.params.devices.push_back(dev);
      }
    }
  }

  // check threads number
  if (params.params.cpuparams.n_threads < 0) {
    params.params.cpuparams.n_threads = cpu_get_num_math();
  }

  if (params.params.cpuparams_batch.n_threads < 0) {
    params.params.cpuparams_batch.n_threads = cpu_get_num_math();
  }

  // models
  if (params.params.model.path.empty()) {
    params.params.model.path = download_model(params.params.model.hf_repo,
                                              params.params.model.hf_file);
  }

  if (params.params.mmproj.path.empty()) {
    params.params.mmproj.path = download_model(params.params.mmproj.hf_repo,
                                               params.params.mmproj.hf_file);
  }

  // lora_adapters
  if (!lora_adapters.empty()) {
    if (lora_adapters.size() != lora_adapters_scales.size()) {
      RCLCPP_ERROR(
          node->get_logger(),
          "lora_adapters and lora_adapters_scales must have the same size");

    } else {

      while (!lora_adapters.empty()) {

        // get lora
        std::string lora = lora_adapters.front();
        lora_adapters.erase(lora_adapters.begin());

        // get scale
        float scale = (float)lora_adapters_scales.front();
        lora_adapters_scales.erase(lora_adapters_scales.begin());

        // check if lora is from HF
        if (lora == "HF") {
          if (lora_adapters_repos.empty() || lora_adapters_filenames.empty()) {
            RCLCPP_ERROR(node->get_logger(),
                         "lora_adapters_repos and lora_adapters_filenames "
                         "must have the same size");
            continue;
          }

          std::string repo = lora_adapters_repos.front();
          std::string filename = lora_adapters_filenames.front();

          lora_adapters_repos.erase(lora_adapters_repos.begin());
          lora_adapters_filenames.erase(lora_adapters_filenames.begin());

          lora = download_model(repo, filename);
        }

        if (lora.empty()) {
          continue;
        }

        // fix scale
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

        // add lora
        params.params.lora_adapters.push_back({lora, scale, nullptr});
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
  if (params.params.pooling_type == LLAMA_POOLING_TYPE_RANK) {
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

common_grammar_trigger_type llama_utils::parse_grammar_trigger_type(int type) {
  if (type == llama_msgs::msg::GrammarTrigger::GRAMMAR_TRIGGER_TYPE_WORD) {
    return COMMON_GRAMMAR_TRIGGER_TYPE_WORD;
  } else if (type ==
             llama_msgs::msg::GrammarTrigger::GRAMMAR_TRIGGER_TYPE_TOKEN) {
    return COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
  } else if (type ==
             llama_msgs::msg::GrammarTrigger::GRAMMAR_TRIGGER_TYPE_PATTERN) {
    return COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN;
  } else if (type == llama_msgs::msg::GrammarTrigger::
                         GRAMMAR_TRIGGER_TYPE_PATTERN_START) {
    return COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL;
  } else {
    throw std::runtime_error("Unsupported grammar trigger type: " +
                             std::to_string(type));
  }
}

struct common_params_sampling llama_utils::parse_sampling_params(
    const llama_msgs::msg::SamplingConfig &sampling_config, int n_vocab) {

  struct common_params_sampling sparams;

  sparams.n_prev = sampling_config.n_prev;
  sparams.n_probs = sampling_config.n_probs;
  sparams.min_keep = sampling_config.min_keep;

  sparams.ignore_eos = sampling_config.ignore_eos;
  for (auto logit_bias : sampling_config.logit_bias.data) {
    sparams.logit_bias.push_back({logit_bias.token, logit_bias.bias});
  }

  sparams.temp = sampling_config.temp;
  sparams.dynatemp_range = sampling_config.dynatemp_range;
  sparams.dynatemp_exponent = sampling_config.dynatemp_exponent;

  sparams.top_k = sampling_config.top_k;
  sparams.top_p = sampling_config.top_p;
  sparams.min_p = sampling_config.min_p;
  sparams.top_n_sigma = sampling_config.top_n_sigma;
  sparams.xtc_probability = sampling_config.xtc_probability;
  sparams.xtc_threshold = sampling_config.xtc_threshold;
  sparams.typ_p = sampling_config.typical_p;

  sparams.penalty_last_n = sampling_config.penalty_last_n;
  sparams.penalty_repeat = sampling_config.penalty_repeat;
  sparams.penalty_freq = sampling_config.penalty_freq;
  sparams.penalty_present = sampling_config.penalty_present;

  sparams.dry_multiplier = sampling_config.dry_multiplier;
  sparams.dry_base = sampling_config.dry_base;
  sparams.dry_allowed_length = sampling_config.dry_allowed_length;
  sparams.dry_penalty_last_n = sampling_config.dry_penalty_last_n;
  sparams.dry_sequence_breakers = sampling_config.dry_sequence_breakers;

  sparams.mirostat = sampling_config.mirostat;
  sparams.mirostat_eta = sampling_config.mirostat_eta;
  sparams.mirostat_tau = sampling_config.mirostat_tau;

  // grammar params
  sparams.samplers =
      common_sampler_types_from_chars(sampling_config.samplers_sequence);
  sparams.grammar = sampling_config.grammar;
  sparams.grammar_lazy = sampling_config.grammar_lazy;

  for (auto grammar_trigger : sampling_config.grammar_triggers) {
    struct common_grammar_trigger trigger;
    trigger.token = grammar_trigger.token;
    trigger.type =
        llama_utils::parse_grammar_trigger_type(grammar_trigger.type);
    trigger.value = grammar_trigger.value;
    sparams.grammar_triggers.push_back(trigger);
  }

  sparams.preserved_tokens =
      std::set<llama_token>(sampling_config.preserved_tokens.begin(),
                            sampling_config.preserved_tokens.end());

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

  return sparams;
}
