// MIT License
//
// Copyright (c) 2023 Miguel Ángel González Santamarta
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

#ifndef LLAMA_ROS__LLAMA_HPP
#define LLAMA_ROS__LLAMA_HPP

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>
#include <future>
#include <queue>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include <mtmd.h>

#include "llama_utils/spinner.hpp"
#include "llama_ros/slot_manager.hpp"
#include "llama_ros/task_registry.hpp"
#include "llama_ros/request_handler.hpp"
#include "llama_ros/result.hpp"
#include "llama_utils/chat_formatter.hpp"

using json = nlohmann::ordered_json;

// Forward declarations to avoid circular dependencies
namespace llama_utils {
  struct ChatCompletionsContext;
}

namespace llama_ros {

struct OAICompactParserOptions {
  bool use_jinja;
  bool prefill_assistant;
  common_reasoning_format reasoning_format;
  std::map<std::string, std::string> chat_template_kwargs;
  common_chat_templates *tmpls;
  bool allow_image;
  bool allow_audio;
  bool enable_thinking = true;
};

/**
 * @brief Represents the probability of a token.
 */
struct TokenProb {
  /**
   * @brief The token.
   */
  llama_token token;

  /**
   * @brief The probability of the token.
   */
  float probability;
};

/**
 * @brief Represents a Low-Rank Adaptation (LoRA) configuration.
 */
struct LoRA {
  /**
   * @brief The ID of the LoRA configuration.
   */
  int id;

  /**
   * @brief The file path to the LoRA model.
   */
  std::string path;

  /**
   * @brief The scaling factor for the LoRA model.
   */
  float scale;
};

/**
 * @brief Represents the output of a completion operation.
 */
struct CompletionOutput {
  /**
   * @brief The probabilities of tokens in the completion.
   */
  std::vector<TokenProb> probs;

  /**
   * @brief The token generated in the completion.
   */
  llama_token token;
  std::string text_to_send;
};

/**
 * @brief Represents the stopping condition for a process.
 */
enum StopType {
  NO_STOP,      /**< @brief No stopping condition. */
  FULL_STOP,    /**< @brief Full stop condition. */
  PARTIAL_STOP, /**< @brief Partial stop condition. */
  CANCEL,       /**< @brief Cancel the process. */
  ABORT         /**< @brief Abort the process. */
};

/**
 * @brief Represents the output of a response generation process.
 */
struct ResponseOutput {
  /**
   * @brief The list of completion outputs.
   */
  std::vector<CompletionOutput> completions;

  /**
   * @brief The stopping condition for the response generation.
   */
  StopType stop;
};

/**
 * @brief Represents the output of an embedding generation process.
 */
struct EmbeddingsOutput {
  /**
   * @brief The generated embeddings.
   */
  std::vector<float> embeddings;

  /**
   * @brief The number of tokens used to generate the embeddings.
   */
  int32_t n_tokens;
};

/**
 * @brief A structure representing the metadata of a model.
 */
struct Metadata {
  /**
   * @brief General information about the model.
   */
  struct GeneralInfo {
    /**
     * @brief The architecture of the model.
     */
    std::string architecture;

    /**
     * @brief The quantization version of the model.
     */
    uint32_t quantization_version;

    /**
     * @brief The alignment of the model.
     */
    uint32_t alignment;

    /**
     * @brief The name of the model.
     */
    std::string name;

    /**
     * @brief The author of the model.
     */
    std::string author;

    /**
     * @brief The version of the model.
     */
    std::string version;

    /**
     * @brief The organization associated with the model.
     */
    std::string organization;

    /**
     * @brief The base name of the model file.
     */
    std::string basename;

    /**
     * @brief The fine-tuning information of the model.
     */
    std::string finetune;

    /**
     * @brief A description of the model.
     */
    std::string description;

    /**
     * @brief The entity that quantized the model.
     */
    std::string quantized_by;

    /**
     * @brief The size label of the model.
     */
    std::string size_label;

    /**
     * @brief The license type of the model.
     */
    std::string license;

    /**
     * @brief The name of the license.
     */
    std::string license_name;

    /**
     * @brief The link to the license.
     */
    std::string license_link;

    /**
     * @brief The URL of the model.
     */
    std::string url;

    /**
     * @brief The repository URL of the model.
     */
    std::string repo_url;

    /**
     * @brief The DOI (Digital Object Identifier) of the model.
     */
    std::string doi;

    /**
     * @brief The UUID (Universally Unique Identifier) of the model.
     */
    std::string uuid;

    /**
     * @brief Tags associated with the model.
     */
    std::vector<std::string> tags;

    /**
     * @brief Languages supported by the model.
     */
    std::vector<std::string> languages;

    /**
     * @brief Datasets used to train the model.
     */
    std::vector<std::string> datasets;

    /**
     * @brief The file type of the model.
     */
    std::string file_type;
  };

  /**
   * @brief A structure representing the attention information of a model.
   */
  struct AttentionInfo {
    /**
     * @brief The number of attention heads.
     */
    uint64_t head_count;

    /**
     * @brief The number of key-value attention heads.
     */
    uint64_t head_count_kv;

    /**
     * @brief The maximum alibi bias.
     */
    float max_alibi_bias;

    /**
     * @brief The clamp value for key-query-value operations.
     */
    float clamp_kqv;

    /**
     * @brief The epsilon value for layer normalization.
     */
    float layer_norm_epsilon;

    /**
     * @brief The epsilon value for RMS layer normalization.
     */
    float layer_norm_rms_epsilon;

    /**
     * @brief The length of the key vector.
     */
    uint32_t key_length;

    /**
     * @brief The length of the value vector.
     */
    uint32_t value_length;
  };

  /**
   * @brief A structure representing the RoPE (Rotary Positional Encoding) of a
   * model.
   */
  struct RoPEInfo {
    /**
     * @brief The number of dimensions used in RoPE.
     */
    uint64_t dimension_count;

    /**
     * @brief The base frequency for RoPE.
     */
    float freq_base;

    /**
     * @brief The scaling type used in RoPE.
     */
    std::string scaling_type;

    /**
     * @brief The scaling factor for RoPE.
     */
    float scaling_factor;

    /**
     * @brief The original context length for scaling.
     */
    uint32_t scaling_original_context_length;

    /**
     * @brief Indicates whether the model was fine-tuned with scaling.
     */
    bool scaling_finetuned;
  };

  /**
   * @brief A structure representing the model information.
   */
  struct ModelInfo {
    /**
     * @brief The context length of the model.
     */
    uint64_t context_length;

    /**
     * @brief The embedding length of the model.
     */
    uint64_t embedding_length;

    /**
     * @brief The number of blocks in the model.
     */
    uint64_t block_count;

    /**
     * @brief The feed-forward length of the model.
     */
    uint64_t feed_forward_length;

    /**
     * @brief Indicates whether parallel residual connections are used.
     */
    bool use_parallel_residual;

    /**
     * @brief The data layout of the model's tensors.
     */
    std::string tensor_data_layout;

    /**
     * @brief The number of experts in the model.
     */
    uint32_t expert_count;

    /**
     * @brief The number of experts used in the model.
     */
    uint32_t expert_used_count;

    /**
     * @brief The attention information of the model.
     */
    AttentionInfo attention;

    /**
     * @brief The RoPE (Rotary Positional Encoding) information of the model.
     */
    RoPEInfo rope;
  };

  /**
   * @brief A structure representing the tokenizer information.
   */
  struct TokenizerInfo {
    /**
     * @brief The tokenizer model used.
     */
    std::string model;

    /**
     * @brief The ID of the beginning-of-sequence (BOS) token.
     */
    uint32_t bos_token_id;

    /**
     * @brief The ID of the end-of-sequence (EOS) token.
     */
    uint32_t eos_token_id;

    /**
     * @brief The ID of the unknown token.
     */
    uint32_t unknown_token_id;

    /**
     * @brief The ID of the padding token.
     */
    uint32_t padding_token_id;

    /**
     * @brief The ID of the separator token.
     */
    uint32_t separator_token_id;

    /**
     * @brief Indicates whether a BOS token is added.
     */
    bool add_bos_token;

    /**
     * @brief The chat template used for tokenization.
     */
    std::string chat_template;
  };

  /**
   * @brief General information about the model.
   */
  GeneralInfo general;

  /**
   * @brief Detailed information about the model.
   */
  ModelInfo model;

  /**
   * @brief Information about the tokenizer used by the model.
   */
  TokenizerInfo tokenizer;
};

enum SlotState {
  SLOT_STATE_IDLE,
  SLOT_STATE_STARTED,
  SLOT_STATE_PROCESSING_PROMPT,
  SLOT_STATE_DONE_PROMPT,
  SLOT_STATE_GENERATING
};

enum ServerTaskType {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_CANCEL,
};


/**
 * @brief Represents a log probability for a token.
 */
struct LogProb {
  /**
   * @brief The token ID.
   */
  int token;

  /**
   * @brief The log probability of the token.
   */
  float probability;

  /**
   * @brief The text representation of the token.
   */
  std::string text;
};

/**
 * @brief Represents a selected log probability and its associated data.
 */
struct SelectedLogProb {
  /**
   * @brief The chosen token and its log probability.
   */
  LogProb chosen_token;

  /**
   * @brief A list of log probabilities for other tokens.
   */
  std::vector<LogProb> data;
};

struct ServerTaskResult {
  uint64_t id;
  int id_slot;
  virtual ~ServerTaskResult() = default;
};

struct ServerTaskResultEmbedding : ServerTaskResult {
  std::vector<std::vector<float>> embeddings;
  int32_t n_tokens;
};

struct ServerTaskResultRerank : ServerTaskResult {
    float score = -1e6;
};

struct ServerTaskResultCompletionPartial : ServerTaskResult {
    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;

    TokenProb prob_output;
    std::string build_info;
    llama_ros::StopType stop;
    bool post_sampling_probs;

    std::string     oaicompat_model;
    std::string     oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs;
};

struct ServerTaskResultCompletion : ServerTaskResult {
    /**
   * @brief The content of the chat response.
   */
  std::string content;

  /**
   * @brief The list of token IDs in the response.
   */
  std::vector<int> tokens;

  /**
   * @brief Indicates if the response is streamed.
   */
  bool stream;

  /**
   * @brief The prompt used to generate the response.
   */
  std::string prompt;

  /**
   * @brief Build information for debugging purposes.
   */
  std::string build_info;

  /**
   * @brief The number of tokens decoded in the response.
   */
  int32_t n_decoded;

  /**
   * @brief The number of tokens in the prompt.
   */
  int32_t n_prompt_tokens;

  /**
   * @brief The stop condition for the response generation.
   */
  llama_ros::StopType stop;

  /**
   * @brief Indicates if post-sampling probabilities are included.
   */
  bool post_sampling_probs;

  /**
   * @brief The output probabilities for selected tokens.
   */
  std::vector<SelectedLogProb> probs_output;

  /**
   * @brief Additional fields included in the response.
   */
  std::vector<std::string> response_fields;

  /**
   * @brief The OpenAI-compatible chat format.
   */
  common_chat_format oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

  /**
   * @brief The OpenAI-compatible model name.
   */
  std::string oaicompat_model;

  /**
   * @brief The OpenAI-compatible completion ID.
   */
  std::string oaicompat_cmpl_id;

  /**
   * @brief The OpenAI-compatible chat syntax. Used while streaming the
   * response.
   */
  common_chat_msg oaicompat_msg;
  std::vector<common_chat_msg_diff> oaicompat_msg_diffs;
};

using ServerTaskResultPtr = std::unique_ptr<ServerTaskResult>;

class ServerSlot {
public:
  /**
 * @brief A callback function type for handling generated responses.
 */
  using GenerateResponseCallback = std::function<void(struct CompletionOutput, ServerSlot *)>;

  int id;
  uint64_t goal_id;
  llama_batch batch;
  llama_context *ctx;
  common_sampler *sampler;
  std::vector<common_adapter_lora_info> lora_adapters;

  ServerTaskType task_type = SERVER_TASK_TYPE_COMPLETION;
  llama_token sampled;

  SlotState state = SLOT_STATE_IDLE;
  json json_schema;
  std::string stopping_word;
  bool has_next_token = true;
  bool has_new_line   = false;

  int32_t n_past = 0;
  int32_t n_ctx = 0;
  int32_t n_consumed = 0;
  int32_t n_predict = -1;
  int32_t i_batch = -1;
  int32_t n_decoded = 0;
  int32_t ga_i = 0;  

  size_t n_sent_text        = 0;
  bool stream;
  GenerateResponseCallback stream_callback = nullptr;

  struct slot_params {
    int32_t n_keep = 0;
    int32_t n_discard = 0;
    int32_t n_predict = -1;
    int32_t n_indent  =  0;

    std::vector<common_adapter_lora_info> lora;

    std::vector<std::string> antiprompt;

    struct common_params_sampling sampling;

    bool verbose = false;
    std::string oaicompat_model;
    std::string oaicompat_cmpl_id;
    common_chat_parser_params oaicompat_chat_syntax;
  } params;

  std::vector<llama_token> prompt_tokens;
  int32_t n_prompt_tokens           = 0;
  int32_t n_prompt_tokens_processed = 0;
  size_t last_nl_pos = 0;

  common_chat_msg chat_msg;
  common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
  std::vector<std::string> generated_tool_call_ids;
  StopType stop;
  std::string generated_text;
  llama_tokens generated_tokens;
  llama_perf_context_data prev_stat_usage;

  std::unordered_map<llama_pos, mtmd::input_chunk_ptr> map_pos_to_media;

  void reset();
  const common_chat_msg &update_chat_msg(std::vector<common_chat_msg_diff> &diffs);
  void release();
  inline bool is_processing() const { return state != SLOT_STATE_IDLE; }
  size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop);
};

/**
 * @brief A class representing a llama.cpp.
 *
 * This class provides methods for tokenization, detokenization,
 * generating responses, and managing the model's state.
 * It also provides methods for managing LoRA models and
 * generating embeddings.
 */
class Llama {

public:
  /**
   * @brief Common parameters for the model.
   *
   * This structure contains configuration parameters used to initialize and
   * manage the model.
   */
  struct common_params params;

  /**
   * @brief Constructor for the Llama class.
   *
   * This constructor initializes the Llama object with the given parameters,
   *
   * @param params The common parameters for the model.
   * @param system_prompt The system prompt to be used.
   * @param initial_reset Whether to reset the model initially.
   */
  Llama(const struct common_params &params, std::string system_prompt = "",
        bool initial_reset = true);

  /**
   * @brief Destructor for the Llama class.
   */
  virtual ~Llama();

  bool process_token(ServerSlot *slot, CompletionOutput *result);
  void run_loop();
  virtual bool process_mtmd_chunk(llama_ros::ServerSlot *slot);

  /**
   * @brief Tokenizes the given text into a vector of tokens.
   *
   * @param text The input text to tokenize.
   * @param add_bos Whether to add a beginning-of-sequence (BOS) token.
   * @param special Whether to include special tokens.
   * @return A vector of tokens representing the tokenized text.
   */
  std::vector<llama_token> tokenize(const std::string &text, bool add_bos,
                                    bool special = false);

  /**
   * @brief Converts a vector of tokens back into a string.
   *
   * @param tokens The vector of tokens to detokenize.
   * @return The detokenized string.
   */
  std::string detokenize(const std::vector<llama_token> &tokens);

  /**
   * @brief Resets the internal state of the model.
   *
   * This method is virtual and can be overridden by derived classes.
   */
  virtual void reset();

  /**
   * @brief Cancels any ongoing operations or computations.
   */
  void cancel();

  /**
   * @brief Cancels a specific goal by its ID.
   *
   * @param goal_id The ID of the goal to cancel.
   */
  void cancel_goal(uint64_t goal_id);

  /**
   * @brief Lists all available LoRA (Low-Rank Adaptation) models.
   *
   * @return A vector of LoRA structures representing the available models.
   */
  std::vector<struct LoRA> list_loras();

  /**
   * @brief Updates the current LoRA models with the provided list.
   *
   * @param loras A vector of LoRA structures to update the models.
   */
  void update_loras(std::vector<struct LoRA> loras);

  /**
   * @brief Generates embeddings for a given input prompt.
   *
   * @param input_prompt The input text prompt for which embeddings are
   * generated.
   * @param normalization The normalization method to apply (default is 2).
   * @return A Result containing embeddings and token count, or an error message.
   */
  Result<ServerTaskResultEmbedding> generate_embeddings(const std::string &text);

  void handle_rerank_req(const std::string &query, const std::string &document, ServerSlot *slot);
  void handle_embeddings_req(const std::string &input_prompt, ServerSlot *slot);
  virtual void handle_completion_req(const std::string &input_prompt, ServerSlot *slot, struct common_params_sampling sparams, ServerSlot::GenerateResponseCallback callback, std::vector<std::string> stop, bool reset);
  virtual void handle_chat_completion_req(llama_utils::ChatCompletionsContext chat_context, ServerSlot *slot, ServerSlot::GenerateResponseCallback callback);

  std::vector<llama_token>
  truncate_tokens(const std::vector<llama_token> &tokens, int limit_size,
                  bool add_eos = true);

  /**
   * @brief Ranks the relevance of multiple documents to a given query.
   *
   * @param query The query string.
   * @param documents A vector of document strings to rank.
   * @return A Result containing relevance scores, or an error message.
   */
  Result<std::vector<llama_ros::ServerTaskResultRerank>> rank_documents(const std::string &query,
                                    const std::vector<std::string> &documents);

  /**
   * @brief Generates a response based on the input prompt and sampling
   * parameters.
   *
   * @param input_prompt The input text prompt for generating the response.
   * @param sparams The sampling parameters to guide the response generation.
   * @param callback (Optional) A callback function to handle the generated
   * response.
   * @param stop (Optional) A list of stop words or phrases to terminate the
   * response generation.
   * @return A Result containing the generated response and metadata, or an error.
   */
  Result<ServerTaskResultCompletion>
  generate_response(int slot_id,
                    const std::string &input_prompt,
                    struct common_params_sampling sparams,
                    ServerSlot::GenerateResponseCallback callback = nullptr,
                    std::vector<std::string> stop = {}, bool reset = true);

  Result<ServerTaskResultCompletion>
  generate_chat_response(int slot_gid,
                          llama_utils::ChatCompletionsContext chat_context,
                          ServerSlot::GenerateResponseCallback callback = nullptr);

  /**
   * @brief Gets the chat formatter utility.
   *
   * @return Pointer to the chat formatter.
   */
  llama_utils::ChatFormatter* get_chat_formatter() {
    return chat_formatter_.get();
  }

  /**
   * @brief Retrieves the chat parameters based on the provided templates and
   * inputs.
   *
   * @param tmpls The chat templates to use.
   * @param inputs The inputs for the chat templates.
   * @return A structure containing the chat parameters.
   */
  struct common_chat_params
  get_chat_params(struct common_chat_templates *tmpls,
                  struct common_chat_templates_inputs inputs);

  /**
   * @brief Retrieves performance context data for the model.
   *
   * @return A structure containing performance context data.
   */
  struct llama_perf_context_data get_perf_data();

  /**
   * @brief Retrieves the internal llama context.
   *
   * @return A pointer to the llama context structure.
   */
  const struct llama_context *get_ctx() { return this->ctx; }

  /**
   * @brief Retrieves the internal llama model.
   *
   * @return A pointer to the llama model structure.
   */
  const struct llama_model *get_model() { return this->model; }

  /**
   * @brief Retrieves the internal llama memory.
   *
   * @return A llama memory.
   */
  llama_memory_t get_memory() { return llama_get_memory(this->ctx); }

  /**
   * @brief Retrieves the vocabulary associated with the llama model.
   *
   * @return A pointer to the llama vocabulary structure.
   */
  const struct llama_vocab *get_vocab() {
    return llama_model_get_vocab(this->model);
  }

  /**
   * @brief Retrieves the context size of the model.
   *
   * @return The number of context tokens supported by the model.
   */
  int get_n_ctx() { return llama_n_ctx(this->ctx); }

  /**
   * @brief Retrieves the training context size of the model.
   *
   * @return The number of context tokens used during training.
   */
  int get_n_ctx_train() { return llama_model_n_ctx_train(this->model); }

  /**
   * @brief Retrieves the embedding size of the model.
   *
   * @return The number of dimensions in the embedding space.
   */
  int get_n_embd() { return llama_model_n_embd(this->model); }

  /**
   * @brief Retrieves the vocabulary size of the model.
   *
   * @return The number of tokens in the model's vocabulary.
   */
  int get_n_vocab() { return llama_vocab_n_tokens(this->get_vocab()); }

  /**
   * @brief Retrieves metadata as a string based on a key.
   *
   * @param key The key for the metadata to retrieve.
   * @param size The size of the metadata value.
   * @return The metadata value as a string.
   */
  std::string get_metadata(const std::string &key, size_t size);

  /**
   * @brief Retrieves metadata as a string for a specific model based on a key.
   *
   * @param model_name The name of the model.
   * @param key The key for the metadata to retrieve.
   * @param size The size of the metadata value.
   * @return The metadata value as a string.
   */
  std::string get_metadata(const std::string &model_name,
                           const std::string &key, size_t size);

  /**
   * @brief Retrieves metadata as an integer based on a key.
   *
   * @param key The key for the metadata to retrieve.
   * @param size The size of the metadata value.
   * @return The metadata value as an integer.
   */
  int get_int_metadata(const std::string &key, size_t size);

  /**
   * @brief Retrieves metadata as an integer for a specific model based on a
   * key.
   *
   * @param model_name The name of the model.
   * @param key The key for the metadata to retrieve.
   * @param size The size of the metadata value.
   * @return The metadata value as an integer.
   */
  int get_int_metadata(const std::string &model_name, const std::string &key,
                       size_t size);

  /**
   * @brief Retrieves metadata as a floating-point value based on a key.
   *
   * @param key The key for the metadata to retrieve.
   * @param size The size of the metadata value.
   * @return The metadata value as a floating-point number.
   */
  float get_float_metadata(const std::string &key, size_t size);

  /**
   * @brief Retrieves metadata as a floating-point value for a specific model
   * based on a key.
   *
   * @param model_name The name of the model.
   * @param key The key for the metadata to retrieve.
   * @param size The size of the metadata value.
   * @return The metadata value as a floating-point number.
   */
  float get_float_metadata(const std::string &model_name,
                           const std::string &key, size_t size);

  /**
   * @brief Retrieves the full metadata structure for the model.
   *
   * @return A structure containing all metadata information.
   */
  struct Metadata get_metadata();

  /**
   * @brief Checks if the model is in embedding mode.
   *
   * @return True if the model is in embedding mode, false otherwise.
   */
  bool is_embedding() { return this->params.embedding; }

  /**
   * @brief Checks if the model is in reranking mode.
   *
   * @return True if the model is in reranking mode, false otherwise.
   */
  bool is_reranking() {
    return this->params.pooling_type == LLAMA_POOLING_TYPE_RANK;
  }

  /**
   * @brief Checks if the model adds a beginning-of-sequence (BOS) token.
   *
   * @return True if the BOS token is added, false otherwise.
   */
  bool add_bos_token() { return llama_vocab_get_add_bos(this->get_vocab()); }

  /**
   * @brief Checks if the model has reached the end-of-generation (EOG).
   *
   * @return True if the end-of-generation is reached, false otherwise.
   */
  bool is_eog(ServerSlot *slot = nullptr) {
    return llama_vocab_is_eog(this->get_vocab(),
                              common_sampler_last(slot->sampler));
  }

  /**
   * @brief Retrieves the end-of-sequence (EOS) token.
   *
   * @return The EOS token.
   */
  llama_token get_token_eos() { return llama_vocab_eos(this->get_vocab()); }

  /**
   * @brief Retrieves the beginning-of-sequence (BOS) token.
   *
   * @return The BOS token.
   */
  llama_token get_token_bos() { return llama_vocab_bos(this->get_vocab()); }

  /**
   * @brief Retrieves the separator token.
   *
   * @return The separator token.
   */
  llama_token get_token_sep() { return llama_vocab_sep(this->get_vocab()); }

  ServerSlot *get_available_slot();
  ServerSlot *wait_for_available_slot();
  ServerSlot *get_slot_by_id(int id);
  ServerSlot *get_slot_by_gid(uint64_t gid);

protected:
  /**
   * @brief Initialization result for the model.
   *
   * This structure holds the result of the model initialization process.
   */
  std::unique_ptr<common_init_result> llama_init;

  /**
   * @brief Pointer to the llama context.
   *
   * This context is used for managing the state and operations of the model.
   */
  struct llama_context *ctx;

  /**
   * @brief Pointer to the llama model.
   *
   * This represents the loaded model used for inference and other operations.
   */
  struct llama_model *model;

  /**
   * @brief List of LoRA (Low-Rank Adaptation) adapters.
   *
   * These adapters are used to modify the behavior of the model.
   */
  std::vector<common_adapter_lora_info> lora_adapters;

  /**
   * @brief Pointer to the sampler used for token sampling.
   *
   * The sampler is responsible for selecting tokens during generation.
   */
  struct common_sampler *sampler;

  llama_batch batch;

  /**
   * @brief Pointer to the thread pool for parallel processing.
   *
   * This thread pool is used for managing tasks during model execution.
   */
  struct ggml_threadpool *threadpool;

  /**
   * @brief Pointer to the thread pool for batch processing.
   *
   * This thread pool is used for managing batch operations during model
   * execution.
   */
  struct ggml_threadpool *threadpool_batch;

  /**
   * @brief The system prompt used for initializing the model's context.
   *
   * This prompt provides context or instructions for the model.
   */
  std::string system_prompt;

  /**
   * @brief Indicates whether the model's operations have been canceled.
   *
   * If true, ongoing operations will be interrupted.
   */
  bool canceled;

  /**
   * @brief Spinner utility for managing asynchronous operations.
   *
   * This spinner is used to indicate progress or manage waiting states.
   */
  llama_utils::Spinner spinner;

  /**
   * @brief Number of past tokens processed by the model.
   *
   * This value is used to manage the model's context window.
   */
  int32_t n_past;

  /**
   * @brief Number of tokens consumed during processing.
   *
   * This value tracks the progress of token consumption.
   */
  int32_t n_consumed;

  /**
   * @brief Internal counter for managing generation steps.
   *
   * This counter is used for tracking the generation process.
   */
  int32_t ga_i;

  std::vector<ServerSlot> server_slots;
  std::unique_ptr<SlotManager> slot_manager_;
  std::unique_ptr<TaskRegistry> task_registry_;
  std::unique_ptr<llama_utils::ChatFormatter> chat_formatter_;
  
  // Request handlers
  std::unique_ptr<EmbeddingRequestHandler> embedding_handler_;
  std::unique_ptr<RerankRequestHandler> rerank_handler_;
  std::unique_ptr<CompletionRequestHandler> completion_handler_;
  std::unique_ptr<ChatCompletionRequestHandler> chat_completion_handler_;

  void release_slot(ServerSlot *slot);

  std::future<ServerTaskResultPtr> register_pending(uint64_t goal_id);
  void fulfill_pending(uint64_t goal_id, ServerTaskResultPtr r);
  void fail_pending(uint64_t goal_id, std::string err);

  void send_embedding_result(ServerSlot *slot, const llama_batch &batch);
  void send_rerank_result(ServerSlot *slot, const llama_batch &batch);
  void send_completion_result(ServerSlot *slot);

  /**
   * @brief Retrieves the probabilities of the next tokens.
   *
   * @return A vector of token probabilities.
   */
  std::vector<struct TokenProb> get_probs(ServerSlot *slot);

  OAICompactParserOptions oai_parser_opt;
};

} // namespace llama_ros

#endif
