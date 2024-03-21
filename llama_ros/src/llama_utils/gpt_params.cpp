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

#include "llama_utils/gpt_params.hpp"

using namespace llama_utils;

GptParams::GptParams() : debug(false) {
  this->params = std::make_shared<struct gpt_params>();
}

std::shared_ptr<struct gpt_params> GptParams::load_params(rclcpp::Node *node) {

  std::string stop;
  std::string file_path;
  std::string lora_adapter;

  std::string split_mode;
  std::string rope_scaling_type;
  std::string numa;
  std::string pooling_type;

  std::vector<double> tensor_split;

  // llama params
  node->declare_parameters<int32_t>("", {
                                            {"seed", -1},
                                            {"n_ctx", 512},
                                            {"n_batch", 2048},
                                            {"n_ubatch", 512},
                                            {"n_gpu_layers", 0},
                                            {"main_gpu", 0},
                                            {"n_threads", 1},
                                            {"n_threads_batch", -1},
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
                                                {"lora_adapter", ""},
                                                {"lora_base", ""},
                                                {"mmproj", ""},
                                                {"split_mode", "layer"},
                                                {"rope_scaling_type", ""},
                                                {"numa", "none"},
                                                {"pooling_type", ""},
                                                {"cache_type_k", "f16"},
                                                {"cache_type_v", "f16"},
                                                {"prompt", ""},
                                                {"file", ""},
                                                {"prefix", ""},
                                                {"suffix", ""},
                                                {"stop", ""},
                                            });
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
                                         {"embedding", true},
                                         {"logits_all", false},
                                         {"use_mmap", true},
                                         {"use_mlock", false},
                                         {"cont_batching", false},
                                         {"dump_kv_cache", false},
                                         {"no_kv_offload", false},
                                     });

  node->get_parameter("seed", this->params->seed);
  node->get_parameter("n_ctx", this->params->n_ctx);
  node->get_parameter("n_batch", this->params->n_batch);
  node->get_parameter("n_ubatch", this->params->n_ubatch);

  node->get_parameter("n_gpu_layers", this->params->n_gpu_layers);
  node->get_parameter("split_mode", split_mode);
  node->get_parameter("main_gpu", this->params->main_gpu);
  node->get_parameter("tensor_split", tensor_split);

  node->get_parameter("embedding", this->params->embedding);
  node->get_parameter("logits_all", this->params->logits_all);
  node->get_parameter("use_mmap", this->params->use_mmap);
  node->get_parameter("use_mlock", this->params->use_mlock);

  node->get_parameter("dump_kv_cache", this->params->dump_kv_cache);
  node->get_parameter("no_kv_offload", this->params->no_kv_offload);
  node->get_parameter("cache_type_k", this->params->cache_type_k);
  node->get_parameter("cache_type_v", this->params->cache_type_v);

  node->get_parameter("n_threads", this->params->n_threads);
  node->get_parameter("n_threads_batch", this->params->n_threads_batch);
  node->get_parameter("n_predict", this->params->n_predict);
  node->get_parameter("n_keep", this->params->n_keep);
  node->get_parameter("n_batch", this->params->n_batch);

  node->get_parameter("grp_attn_n", this->params->grp_attn_n);
  node->get_parameter("grp_attn_w", this->params->grp_attn_w);

  node->get_parameter("rope_freq_base", this->params->rope_freq_base);
  node->get_parameter("rope_freq_scale", this->params->rope_freq_scale);
  node->get_parameter("rope_scaling_type", rope_scaling_type);

  node->get_parameter("yarn_ext_factor", this->params->yarn_ext_factor);
  node->get_parameter("yarn_attn_factor", this->params->yarn_attn_factor);
  node->get_parameter("yarn_beta_fast", this->params->yarn_beta_fast);
  node->get_parameter("yarn_beta_slow", this->params->yarn_beta_slow);
  node->get_parameter("yarn_orig_ctx", this->params->yarn_orig_ctx);

  node->get_parameter("model", this->params->model);
  node->get_parameter("lora_adapter", lora_adapter);
  node->get_parameter("lora_base", this->params->lora_base);
  node->get_parameter("mmproj", this->params->mmproj);
  node->get_parameter("numa", numa);
  node->get_parameter("pooling_type", pooling_type);

  node->get_parameter("n_parallel", this->params->n_parallel);
  node->get_parameter("n_sequences", this->params->n_sequences);
  node->get_parameter("cont_batching", this->params->cont_batching);

  node->get_parameter("prefix", this->params->input_prefix);
  node->get_parameter("suffix", this->params->input_suffix);
  node->get_parameter("stop", stop);

  node->get_parameter("prompt", this->params->prompt);
  node->get_parameter("file", file_path);
  node->get_parameter("debug", this->debug);

  // check threads number
  if (this->params->n_threads < 0) {
    this->params->n_threads = std::thread::hardware_concurrency();
  }

  // lora_adapter
  if (lora_adapter.size()) {
    this->params->lora_adapter.push_back({lora_adapter, 1.0f});
    this->params->use_mmap = false;
  }

  // stop is the antiprompt
  this->params->antiprompt.push_back(stop);

  // split mode
  if (split_mode == "none") {
    this->params->split_mode = LLAMA_SPLIT_MODE_NONE;
  } else if (split_mode == "layer") {
    this->params->split_mode = LLAMA_SPLIT_MODE_LAYER;
  } else if (split_mode == "row") {
    this->params->split_mode = LLAMA_SPLIT_MODE_ROW;
  }

  // rope_scaling_type
  if (rope_scaling_type == "none") {
    this->params->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
  } else if (rope_scaling_type == "linear") {
    this->params->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
  } else if (rope_scaling_type == "yarn") {
    this->params->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
  } else {
    this->params->rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
  }

  // numa
  if (numa == "none") {
    this->params->numa = GGML_NUMA_STRATEGY_DISABLED;
  } else if (numa == "distribute") {
    this->params->numa = GGML_NUMA_STRATEGY_DISTRIBUTE;
  } else if (numa == "isolate") {
    this->params->numa = GGML_NUMA_STRATEGY_ISOLATE;
  } else if (numa == "numactl") {
    this->params->numa = GGML_NUMA_STRATEGY_NUMACTL;
  } else if (numa == "mirror") {
    this->params->numa = GGML_NUMA_STRATEGY_MIRROR;
  }

  // pooling
  if (pooling_type == "none") {
    this->params->pooling_type = LLAMA_POOLING_TYPE_NONE;
  } else if (pooling_type == "mean") {
    this->params->pooling_type = LLAMA_POOLING_TYPE_MEAN;
  } else if (pooling_type == "cls") {
    this->params->pooling_type = LLAMA_POOLING_TYPE_CLS;
  } else {
    this->params->pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
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
              back_inserter(this->params->prompt));
  }

  // split tensors
  GGML_ASSERT(tensor_split.size() <= llama_max_devices());
  for (size_t i = 0; i < llama_max_devices(); ++i) {
    if (i < tensor_split.size()) {
      this->params->tensor_split[i] = tensor_split[i];
    } else {
      this->params->tensor_split[i] = 0.0f;
    }
  }

  return this->params;
}

bool GptParams::update_sampling_params(
    const llama_msgs::msg::SamplingConfig &sampling_config, int n_vocab,
    llama_token token_eos) {

  this->params->sparams.n_prev = sampling_config.n_prev;
  this->params->sparams.n_probs = sampling_config.n_probs;

  this->params->ignore_eos = sampling_config.ignore_eos;

  this->params->sparams.temp = sampling_config.temp;

  this->params->sparams.top_k = sampling_config.top_k;
  this->params->sparams.top_p = sampling_config.top_p;
  this->params->sparams.min_p = sampling_config.min_p;
  this->params->sparams.tfs_z = sampling_config.tfs_z;
  this->params->sparams.typical_p = sampling_config.typical_p;

  this->params->sparams.penalty_last_n = sampling_config.penalty_last_n;
  this->params->sparams.penalty_repeat = sampling_config.penalty_repeat;
  this->params->sparams.penalty_freq = sampling_config.penalty_freq;
  this->params->sparams.penalty_present = sampling_config.penalty_present;

  this->params->sparams.mirostat = sampling_config.mirostat;
  this->params->sparams.mirostat_eta = sampling_config.mirostat_eta;
  this->params->sparams.mirostat_tau = sampling_config.mirostat_tau;

  this->params->sparams.penalize_nl = sampling_config.penalize_nl;

  this->params->sparams.samplers_sequence =
      sampler_types_from_chars(sampling_config.samplers_sequence);
  this->params->sparams.grammar = sampling_config.grammar;

  if (this->params->sparams.grammar.size() == 0 &&
      sampling_config.grammar_schema.size() > 0) {

    this->params->sparams.grammar = json_schema_to_grammar(
        nlohmann::json::parse(sampling_config.grammar_schema));
  }

  // check penalty_last_n
  this->params->sparams.penalty_last_n =
      this->params->sparams.penalty_last_n < 0
          ? this->params->sparams.n_prev
          : this->params->sparams.penalty_last_n;

  // check top_k
  this->params->sparams.top_k =
      this->params->sparams.top_k <= 0 ? n_vocab : this->params->sparams.top_k;

  // add logit bias
  for (auto logit_bias : sampling_config.logit_bias.data) {
    this->params->sparams.logit_bias[logit_bias.token] = logit_bias.bias;
  }

  // add llama_token_eos
  if (params->ignore_eos) {
    this->params->sparams.logit_bias[token_eos] = -INFINITY;
  }

  return true;
}
