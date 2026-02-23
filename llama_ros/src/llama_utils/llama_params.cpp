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
//
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
#include "json.hpp"

#include "huggingface_hub.h"
#include "json-schema-to-grammar.h"
#include "yaml-cpp/yaml.h"

#include "ament_index_cpp/get_package_share_directory.hpp"
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

  // General integer parameters
  node->declare_parameters<int32_t>("", {
                                            {"verbosity", 3},
                                            {"seed", -1},
                                            {"n_ctx", 0},
                                            {"n_batch", 2048},
                                            {"n_ubatch", 512},
                                            {"n_keep", 0},
                                            {"n_chunks", -1},
                                            {"n_predict", -1},
                                            {"n_parallel", 1},
                                            {"n_sequences", 1},
                                            {"n_gpu_layers", -1},
                                            {"main_gpu", 0},
                                        });

  node->declare_parameters<int32_t>("cpu", {
                                               {"n_threads", -1},
                                               {"poll", 50},
                                           });

  node->declare_parameters<int32_t>("cpu_batch", {
                                                     {"n_threads", -1},
                                                     {"poll", 50},
                                                 });

  node->declare_parameters<int32_t>("grp_attn", {
                                                    {"n", 1},
                                                    {"w", 512},
                                                });

  node->declare_parameters<int32_t>("yarn", {
                                                {"orig_ctx", 0},
                                            });

  node->declare_parameters<int32_t>("fit", {
                                               {"min_ctx", 4096},
                                           });

  // General string parameters
  node->declare_parameters<std::string>("", {
                                                {"split_mode", "layer"},
                                                {"numa", "none"},
                                                {"flash_attn_type", "auto"},
                                                {"pooling_type", ""},
                                                {"attention_type", ""},
                                                {"prefix", ""},
                                                {"suffix", ""},
                                                {"system_prompt", ""},
                                                {"system_prompt_file", ""},
                                                {"chat_template_file", ""},
                                                {"system_prompt_type", ""},
                                            });

  node->declare_parameters<std::string>("model", {
                                                     {"path", ""},
                                                     {"repo", ""},
                                                     {"filename", ""},
                                                 });

  node->declare_parameters<std::string>("mmproj", {
                                                      {"path", ""},
                                                      {"repo", ""},
                                                      {"filename", ""},
                                                  });

  node->declare_parameters<std::string>("cpu", {
                                                   {"mask", ""},
                                                   {"range", ""},
                                                   {"priority", "normal"},
                                               });

  node->declare_parameters<std::string>("cpu_batch", {
                                                         {"mask", ""},
                                                         {"range", ""},
                                                         {"priority", "normal"},
                                                     });

  node->declare_parameters<std::string>("rope", {
                                                    {"scaling_type", ""},
                                                });

  node->declare_parameters<std::string>("cache", {
                                                     {"type_k", "f16"},
                                                     {"type_v", "f16"},
                                                 });

  node->declare_parameters<std::vector<std::string>>(
      {""}, {
                {"devices", std::vector<std::string>({})},
                {"stopping_words", std::vector<std::string>({})},
                {"loras", std::vector<std::string>({})},
            });

  // RoPE float parameters
  node->declare_parameters<float>("rope", {
                                              {"freq_base", 0.0f},
                                              {"freq_scale", 0.0f},
                                          });

  // Yarn float parameters
  node->declare_parameters<float>("yarn", {
                                              {"ext_factor", -1.0f},
                                              {"attn_factor", -1.0f},
                                              {"beta_fast", -1.0f},
                                              {"beta_slow", -1.0f},
                                          });

  node->declare_parameter<std::vector<double>>("tensor_split",
                                               std::vector<double>({0.0}));

  // General boolean parameters
  node->declare_parameters<bool>("", {
                                         {"embedding", false},
                                         {"reranking", false},
                                         {"use_mmap", true},
                                         {"use_direct_io", false},
                                         {"use_mlock", false},
                                         {"warmup", true},
                                         {"check_tensors", false},
                                         {"ctx_shift", false},
                                         {"swa_full", false},
                                         {"no_op_offload", false},
                                         {"no_extra_bufts", false},
                                         {"no_kv_offload", false},
                                         {"no_host", false},
                                         {"kv_unified", false},
                                         {"cont_batching", true},
                                         {"lora_init_without_apply", false},
                                     });

  node->declare_parameters<bool>("cpu", {
                                            {"strict", false},
                                        });

  node->declare_parameters<bool>("cpu_batch", {
                                                  {"strict", false},
                                              });

  node->declare_parameters<bool>("mmproj", {
                                               {"use_gpu", true},
                                               {"disabled", false},
                                           });

  node->declare_parameters<bool>("fit", {
                                            {"enabled", true},
                                        });
}

LlamaParams llama_utils::get_llama_params(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr &node) {

  int32_t seed;
  int32_t poll;
  int32_t poll_batch;

  bool reranking = false;

  std::vector<std::string> stopping_words;
  std::string chat_template_file;
  std::string system_prompt_type;

  std::vector<std::string> loras;

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
  std::string flash_attn_type;
  std::string pooling_type;
  std::string attention_type;

  std::string system_prompt_file_path;

  LlamaParams params;

  node->get_parameter("verbosity", params.params.verbosity);
  node->get_parameter("seed", seed);
  node->get_parameter("n_ctx", params.params.n_ctx);
  node->get_parameter("n_batch", params.params.n_batch);
  node->get_parameter("n_ubatch", params.params.n_ubatch);
  node->get_parameter("n_keep", params.params.n_keep);
  node->get_parameter("n_chunks", params.params.n_chunks);
  node->get_parameter("n_predict", params.params.n_predict);
  node->get_parameter("n_parallel", params.params.n_parallel);
  node->get_parameter("n_sequences", params.params.n_sequences);

  node->get_parameter("devices", devices);
  node->get_parameter("n_gpu_layers", params.params.n_gpu_layers);
  node->get_parameter("split_mode", split_mode);
  node->get_parameter("main_gpu", params.params.main_gpu);
  node->get_parameter("tensor_split", tensor_split);

  node->get_parameter("embedding", params.params.embedding);
  node->get_parameter("reranking", reranking);
  node->get_parameter("use_mmap", params.params.use_mmap);
  node->get_parameter("use_direct_io", params.params.use_direct_io);
  node->get_parameter("use_mlock", params.params.use_mlock);
  node->get_parameter("warmup", params.params.warmup);
  node->get_parameter("check_tensors", params.params.check_tensors);
  node->get_parameter("ctx_shift", params.params.ctx_shift);
  node->get_parameter("swa_full", params.params.swa_full);

  node->get_parameter("no_op_offload", params.params.no_op_offload);
  node->get_parameter("no_extra_bufts", params.params.no_extra_bufts);
  node->get_parameter("no_kv_offload", params.params.no_kv_offload);
  node->get_parameter("no_host", params.params.no_host);
  node->get_parameter("kv_unified", params.params.kv_unified);
  node->get_parameter("cache.type_k", cache_type_k);
  node->get_parameter("cache.type_v", cache_type_v);

  node->get_parameter("cpu.n_threads", params.params.cpuparams.n_threads);
  node->get_parameter("cpu.mask", cpu_mask);
  node->get_parameter("cpu.range", cpu_range);
  node->get_parameter("cpu.priority", priority);
  node->get_parameter("cpu.strict", params.params.cpuparams.strict_cpu);
  node->get_parameter("cpu.poll", poll);

  node->get_parameter("cpu_batch.n_threads",
                      params.params.cpuparams_batch.n_threads);
  node->get_parameter("cpu_batch.mask", cpu_mask_batch);
  node->get_parameter("cpu_batch.range", cpu_range_batch);
  node->get_parameter("cpu_batch.priority", priority_batch);
  node->get_parameter("cpu_batch.strict",
                      params.params.cpuparams_batch.strict_cpu);
  node->get_parameter("cpu_batch.poll", poll_batch);

  node->get_parameter("grp_attn.n", params.params.grp_attn_n);
  node->get_parameter("grp_attn.w", params.params.grp_attn_w);

  node->get_parameter("rope.freq_base", params.params.rope_freq_base);
  node->get_parameter("rope.freq_scale", params.params.rope_freq_scale);
  node->get_parameter("rope.scaling_type", rope_scaling_type);

  node->get_parameter("yarn.ext_factor", params.params.yarn_ext_factor);
  node->get_parameter("yarn.attn_factor", params.params.yarn_attn_factor);
  node->get_parameter("yarn.beta_fast", params.params.yarn_beta_fast);
  node->get_parameter("yarn.beta_slow", params.params.yarn_beta_slow);
  node->get_parameter("yarn.orig_ctx", params.params.yarn_orig_ctx);

  node->get_parameter("mmproj.use_gpu", params.params.mmproj_use_gpu);
  node->get_parameter("mmproj.disabled", params.params.no_mmproj);

  node->get_parameter("fit.enabled", params.params.fit_params);
  node->get_parameter("fit.min_ctx", params.params.fit_params_min_ctx);

  node->get_parameter("model.path", params.params.model.path);
  node->get_parameter("model.repo", params.params.model.hf_repo);
  node->get_parameter("model.filename", params.params.model.hf_file);
  node->get_parameter("mmproj.path", params.params.mmproj.path);
  node->get_parameter("mmproj.repo", params.params.mmproj.hf_repo);
  node->get_parameter("mmproj.filename", params.params.mmproj.hf_file);

  node->get_parameter("lora_init_without_apply",
                      params.params.lora_init_without_apply);
  node->get_parameter("loras", loras);

  node->get_parameter("numa", numa);
  node->get_parameter("flash_attn_type", flash_attn_type);
  node->get_parameter("pooling_type", pooling_type);
  node->get_parameter("attention_type", attention_type);

  node->get_parameter("cont_batching", params.params.cont_batching);

  node->get_parameter("prefix", params.params.input_prefix);
  node->get_parameter("suffix", params.params.input_suffix);
  node->get_parameter("stopping_words", stopping_words);
  node->get_parameter("system_prompt", params.system_prompt);
  node->get_parameter("system_prompt_file", system_prompt_file_path);
  node->get_parameter("chat_template_file", chat_template_file);
  node->get_parameter("system_prompt_type", system_prompt_type);

  // seed
  if (seed < 0) {
    params.params.sampling.seed = LLAMA_DEFAULT_SEED;
  } else {
    params.params.sampling.seed = seed;
  }

  // Cache type
  params.params.cache_type_k = kv_cache_type_from_str(cache_type_k);
  params.params.cache_type_v = kv_cache_type_from_str(cache_type_v);

  // Devices
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

  // Check threads number
  if (params.params.cpuparams.n_threads < 0) {
    params.params.cpuparams.n_threads = cpu_get_num_math();
  }

  if (params.params.cpuparams_batch.n_threads < 0) {
    params.params.cpuparams_batch.n_threads = cpu_get_num_math();
  }

  // Models
  if (params.params.model.path.empty()) {
    params.params.model.path = download_model(params.params.model.hf_repo,
                                              params.params.model.hf_file);
  }

  if (params.params.mmproj.path.empty()) {
    params.params.mmproj.path = download_model(params.params.mmproj.hf_repo,
                                               params.params.mmproj.hf_file);
  }

  // LoRA adapters
  for (const std::string &lora_name : loras) {

    if (lora_name.empty()) {
      continue;
    }

    // Declare and get per-lora parameters
    if (!node->has_parameter(lora_name + ".repo")) {
      node->declare_parameter<std::string>(lora_name + ".repo", "");
    }
    if (!node->has_parameter(lora_name + ".filename")) {
      node->declare_parameter<std::string>(lora_name + ".filename", "");
    }
    if (!node->has_parameter(lora_name + ".scale")) {
      node->declare_parameter<double>(lora_name + ".scale", 1.0);
    }
    if (!node->has_parameter(lora_name + ".file_path")) {
      node->declare_parameter<std::string>(lora_name + ".file_path", "");
    }

    std::string repo, filename, file_path;
    double scale_d;

    node->get_parameter(lora_name + ".repo", repo);
    node->get_parameter(lora_name + ".filename", filename);
    node->get_parameter(lora_name + ".scale", scale_d);
    node->get_parameter(lora_name + ".file_path", file_path);

    float scale = static_cast<float>(scale_d);

    // Resolve lora path: prefer file_path, then download from HF
    std::string lora_path = file_path;
    if (lora_path.empty() && !repo.empty() && !filename.empty()) {
      lora_path = download_model(repo, filename);
    }

    if (lora_path.empty()) {
      RCLCPP_ERROR(node->get_logger(),
                   "LoRA '%s' has no file_path and no valid repo/filename",
                   lora_name.c_str());
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
    params.params.lora_adapters.push_back({lora_path, scale, "", "", nullptr});
  }

  // Stopping words are the antiprompt
  for (std::string word : stopping_words) {

    if (word.empty()) {
      continue;
    }

    replace_all(word, "\\n", "\n");
    params.params.antiprompt.push_back(word);
  }

  // Initial system prompt
  if (!system_prompt_file_path.empty() && params.system_prompt.empty()) {
    std::ifstream file(system_prompt_file_path.c_str());
    if (!file) {
      RCLCPP_ERROR(node->get_logger(), "Failed to open file %s",
                   system_prompt_file_path.c_str());
    }
    std::copy(std::istreambuf_iterator<char>(file),
              std::istreambuf_iterator<char>(),
              back_inserter(params.system_prompt));
  }

  // Read chat template file if provided
  if (!chat_template_file.empty()) {

    // If the path does not contain "/", prepend the share directory
    if (chat_template_file.find("/") == std::string::npos) {
      chat_template_file =
          ament_index_cpp::get_package_share_directory("llama_cpp_vendor") +
          "/models/templates/" + chat_template_file;
    }

    std::ifstream file(chat_template_file.c_str());
    if (!file) {
      RCLCPP_ERROR(node->get_logger(), "Failed to open chat template file %s",
                   chat_template_file.c_str());
    } else {
      std::copy(std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                back_inserter(params.params.chat_template));
    }
  }

  // Read system prompt type data
  std::string system_prompt_type_file_path =
      ament_index_cpp::get_package_share_directory("llama_ros") + "/prompts/" +
      system_prompt_type + ".yaml";

  if (std::filesystem::exists(system_prompt_type_file_path)) {
    try {
      YAML::Node yaml = YAML::LoadFile(system_prompt_type_file_path);

      if (yaml["prefix"] && params.params.input_prefix.empty()) {
        params.params.input_prefix = yaml["prefix"].as<std::string>();
      }

      if (yaml["suffix"] && params.params.input_suffix.empty()) {
        params.params.input_suffix = yaml["suffix"].as<std::string>();
      }

      if (yaml["stopping_words"]) {
        for (const auto &word : yaml["stopping_words"]) {
          params.params.antiprompt.push_back(word.as<std::string>());
        }
      }

      if (yaml["system_prompt"]) {
        params.system_prompt = yaml["system_prompt"].as<std::string>();
      }
    } catch (const YAML::Exception &e) {
      RCLCPP_ERROR(node->get_logger(),
                   "Failed to parse system prompt type file %s: %s",
                   system_prompt_type_file_path.c_str(), e.what());
    }
  }

  // Split mode
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
  if (reranking) {
    pooling_type = "rerank";
    params.params.embedding = true;
  }

  // rope_scaling_type
  if (rope_scaling_type == "none") {
    params.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
  } else if (rope_scaling_type == "linear") {
    params.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
  } else if (rope_scaling_type == "yarn") {
    params.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
  } else if (rope_scaling_type == "longrope") {
    params.params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LONGROPE;
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

  // flash_attn_type
  if (flash_attn_type == "auto") {
    params.params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
  } else if (flash_attn_type == "enabled") {
    params.params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  } else if (flash_attn_type == "disabled") {
    params.params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  } else {
    params.params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
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
  } else if (pooling_type == "rerank") {
    params.params.pooling_type = LLAMA_POOLING_TYPE_RANK;
  } else {
    params.params.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
  }

  // attention_type
  if (attention_type == "causal") {
    params.params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
  } else if (attention_type == "non_causal") {
    params.params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
  } else {
    params.params.attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;
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
  if (priority == "low") {
    return GGML_SCHED_PRIO_LOW;
  } else if (priority == "normal") {
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

common_params_sampling llama_utils::parse_sampling_params(
    const llama_msgs::msg::SamplingConfig &sampling_config, int n_vocab) {

  common_params_sampling sparams;

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

  sparams.adaptive_target = sampling_config.adaptive_target;
  sparams.adaptive_decay = sampling_config.adaptive_decay;

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

void llama_utils::apply_eog_logit_biases(
    llama_msgs::msg::SamplingConfig &sampling_config,
    const struct llama_vocab *vocab, const struct llama_context *ctx) {

  // Check if vocab has an EOS token when ignore_eos is enabled
  if (sampling_config.ignore_eos &&
      llama_vocab_eos(vocab) == LLAMA_TOKEN_NULL) {
    LLAMA_LOG_WARN("vocab does not have an EOS token, ignoring --ignore-eos\n");
    sampling_config.ignore_eos = false;
  }

  LLAMA_LOG_INFO("Using ignore_eos = %s",
                 sampling_config.ignore_eos ? "true" : "false");

  // Collect all EOG tokens and add them to logit_bias_eog
  for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); i++) {
    if (llama_vocab_is_eog(vocab, i)) {
      LLAMA_LOG_WARN("added %s logit bias = %f\n",
                     common_token_to_piece(ctx, i).c_str(), -INFINITY);
      llama_msgs::msg::LogitBias bias_eog;
      bias_eog.token = i;
      bias_eog.bias = -INFINITY;
      sampling_config.logit_bias_eog.data.push_back(bias_eog);
    }
  }

  LLAMA_LOG_INFO("Using %ld EOG logit biases",
                 sampling_config.logit_bias_eog.data.size());

  // Apply EOG biases to the active logit bias set if ignore_eos is enabled
  if (sampling_config.ignore_eos) {
    sampling_config.logit_bias.data.insert(
        sampling_config.logit_bias.data.end(),
        sampling_config.logit_bias_eog.data.begin(),
        sampling_config.logit_bias_eog.data.end());
  }
}
