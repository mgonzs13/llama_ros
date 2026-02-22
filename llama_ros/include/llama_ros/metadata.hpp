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

#ifndef LLAMA_ROS__METADATA_HPP
#define LLAMA_ROS__METADATA_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace llama_ros {

/**
 * @brief A structure representing the metadata of a model.
 */
struct Metadata {

  /**
   * @brief General information about the model.
   */
  struct GeneralInfo {
    /// @brief The architecture of the model.
    std::string architecture;

    /// @brief The quantization version of the model.
    uint32_t quantization_version;

    /// @brief The alignment of the model.
    uint32_t alignment;

    /// @brief The name of the model.
    std::string name;

    /// @brief The author of the model.
    std::string author;

    /// @brief The version of the model.
    std::string version;

    /// @brief The organization associated with the model.
    std::string organization;

    /// @brief The base name of the model file.
    std::string basename;

    /// @brief The fine-tuning information of the model.
    std::string finetune;

    /// @brief A description of the model.
    std::string description;

    /// @brief The entity that quantized the model.
    std::string quantized_by;

    /// @brief The size label of the model.
    std::string size_label;

    /// @brief The license type of the model.
    std::string license;

    /// @brief The name of the license.
    std::string license_name;

    /// @brief The link to the license.
    std::string license_link;

    /// @brief The URL of the model.
    std::string url;

    /// @brief The repository URL of the model.
    std::string repo_url;

    /// @brief The DOI (Digital Object Identifier) of the model.
    std::string doi;

    /// @brief The UUID (Universally Unique Identifier) of the model.
    std::string uuid;

    /// @brief Tags associated with the model.
    std::vector<std::string> tags;

    /// @brief Languages supported by the model.
    std::vector<std::string> languages;

    /// @brief Datasets used to train the model.
    std::vector<std::string> datasets;

    /// @brief The file type of the model.
    std::string file_type;
  };

  /**
   * @brief Attention information of the model.
   */
  struct AttentionInfo {
    /// @brief The number of attention heads.
    uint64_t head_count;

    /// @brief The number of key-value attention heads.
    uint64_t head_count_kv;

    /// @brief The maximum alibi bias.
    float max_alibi_bias;

    /// @brief The clamp value for key-query-value operations.
    float clamp_kqv;

    /// @brief The epsilon value for layer normalization.
    float layer_norm_epsilon;

    /// @brief The epsilon value for RMS layer normalization.
    float layer_norm_rms_epsilon;

    /// @brief The length of the key vector.
    uint32_t key_length;

    /// @brief The length of the value vector.
    uint32_t value_length;
  };

  /**
   * @brief RoPE (Rotary Positional Encoding) information of a model.
   */
  struct RoPEInfo {
    /// @brief The number of dimensions used in RoPE.
    uint64_t dimension_count;

    /// @brief The base frequency for RoPE.
    float freq_base;

    /// @brief The scaling type used in RoPE.
    std::string scaling_type;

    /// @brief The scaling factor for RoPE.
    float scaling_factor;

    /// @brief The original context length for scaling.
    uint32_t scaling_original_context_length;

    /// @brief Whether the model was fine-tuned with scaling.
    bool scaling_finetuned;
  };

  /**
   * @brief Model architecture information.
   */
  struct ModelInfo {
    /// @brief The context length of the model.
    uint64_t context_length;

    /// @brief The embedding length of the model.
    uint64_t embedding_length;

    /// @brief The number of blocks in the model.
    uint64_t block_count;

    /// @brief The feed-forward length of the model.
    uint64_t feed_forward_length;

    /// @brief Whether parallel residual connections are used.
    bool use_parallel_residual;

    /// @brief The data layout of the model's tensors.
    std::string tensor_data_layout;

    /// @brief The number of experts in the model.
    uint32_t expert_count;

    /// @brief The number of experts used in the model.
    uint32_t expert_used_count;

    /// @brief The attention information of the model.
    AttentionInfo attention;

    /// @brief The RoPE information of the model.
    RoPEInfo rope;
  };

  /**
   * @brief Tokenizer information.
   */
  struct TokenizerInfo {
    /// @brief The tokenizer model used.
    std::string model;

    /// @brief The ID of the beginning-of-sequence (BOS) token.
    uint32_t bos_token_id;

    /// @brief The ID of the end-of-sequence (EOS) token.
    uint32_t eos_token_id;

    /// @brief The ID of the unknown token.
    uint32_t unknown_token_id;

    /// @brief The ID of the padding token.
    uint32_t padding_token_id;

    /// @brief The ID of the separator token.
    uint32_t separator_token_id;

    /// @brief Whether a BOS token is added.
    bool add_bos_token;

    /// @brief The chat template used for tokenization.
    std::string chat_template;
  };

  /// @brief General information about the model.
  GeneralInfo general;

  /// @brief Detailed information about the model.
  ModelInfo model;

  /// @brief Information about the tokenizer used by the model.
  TokenizerInfo tokenizer;
};

} // namespace llama_ros

#endif // LLAMA_ROS__METADATA_HPP
