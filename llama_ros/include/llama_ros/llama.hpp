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

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include "llama_utils/spinner.hpp"

namespace llama_ros {

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
struct EmbeddingsOuput {
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

/**
 * @brief A callback function type for handling generated responses.
 */
using GenerateResponseCallback = std::function<void(struct CompletionOutput)>;

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
   * @brief Truncates a vector of tokens to a specified size.
   *
   * @param tokens The vector of tokens to truncate.
   * @param limit_size The maximum number of tokens to retain.
   * @param add_eos Whether to add an end-of-sequence (EOS) token after
   * truncation.
   * @return A truncated vector of tokens.
   */
  std::vector<llama_token>
  truncate_tokens(const std::vector<llama_token> &tokens, int limit_size,
                  bool add_eos = true);

  /**
   * @brief Generates embeddings for a given input prompt.
   *
   * @param input_prompt The input text prompt for which embeddings are
   * generated.
   * @param normalization The normalization method to apply (default is 2).
   * @return A structure containing the generated embeddings and token count.
   */
  struct EmbeddingsOuput generate_embeddings(const std::string &input_prompt,
                                             int normalization = 2);

  /**
   * @brief Generates embeddings for a given vector of tokens.
   *
   * @param tokens The vector of tokens for which embeddings are generated.
   * @param normalization The normalization method to apply (default is 2).
   * @return A structure containing the generated embeddings and token count.
   */
  struct EmbeddingsOuput
  generate_embeddings(const std::vector<llama_token> &tokens,
                      int normalization = 2);

  /**
   * @brief Ranks the relevance of a document to a given query.
   *
   * @param query The query string.
   * @param document The document string to rank.
   * @return A floating-point score representing the relevance of the document.
   */
  float rank_document(const std::string &query, const std::string &document);

  /**
   * @brief Ranks the relevance of multiple documents to a given query.
   *
   * @param query The query string.
   * @param documents A vector of document strings to rank.
   * @return A vector of floating-point scores representing the relevance of
   * each document.
   */
  std::vector<float> rank_documents(const std::string &query,
                                    const std::vector<std::string> &documents);

  /**
   * @brief Generates a response based on the input prompt and sampling
   * parameters.
   *
   * @param input_prompt The input text prompt for generating the response.
   * @param sparams The sampling parameters to guide the response generation.
   * @param callbakc (Optional) A callback function to handle the generated
   * response.
   * @param stop (Optional) A list of stop words or phrases to terminate the
   * response generation.
   * @return A structure containing the generated response and its metadata.
   */
  struct ResponseOutput
  generate_response(const std::string &input_prompt,
                    struct common_params_sampling sparams,
                    GenerateResponseCallback callbakc = nullptr,
                    std::vector<std::string> stop = {});

  /**
   * @brief Generates a response based on the input prompt.
   *
   * @param input_prompt The input text prompt for generating the response.
   * @param callbakc (Optional) A callback function to handle the generated
   * response.
   * @param stop (Optional) A list of stop words or phrases to terminate the
   * response generation.
   * @return A structure containing the generated response and its metadata.
   */
  struct ResponseOutput
  generate_response(const std::string &input_prompt,
                    GenerateResponseCallback callbakc = nullptr,
                    std::vector<std::string> stop = {});

  /**
   * @brief Retrieves the chat templates used for generating responses.
   *
   * @return A unique pointer to the chat templates structure.
   */
  struct std::unique_ptr<struct common_chat_templates,
                         common_chat_templates_deleter>
  get_chat_templates();

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
  bool is_eog() {
    return llama_vocab_is_eog(this->get_vocab(),
                              common_sampler_last(this->sampler));
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

  /**
   * @brief Updates the chat message based on the specified stop condition and
   * syntax.
   *
   * This method returns a constant reference to the completed chat message.
   * It updates the oaicompat_msg_diffs with the changes made to the chat
   * message while streaming the response.
   *
   * @param stop The stop condition to apply when updating the chat message.
   * @param syntax The syntax rules to use for updating the chat message.
   * @return A constant reference to the completed chat message.
   */
  const common_chat_msg &update_chat_msg(enum StopType stop);

  /**
   * @brief The generated text from the model while streaming the response.
   * @note slot
   */
  std::string generated_text;

  /**
   * @brief Message diffs when streaming the response.
   * @note slot
   */
  std::vector<common_chat_msg_diff> oaicompat_msg_diffs;

  /**
   * @brief The chat syntax used for generating responses.
   * @note slot
   */
  common_chat_syntax oaicompat_chat_syntax;

  /**
   * @brief The previous performance context data for usage statistics. It is
   * used while streaming.
   */
  llama_perf_context_data prev_stat_usage;

protected:
  /**
   * @brief Common parameters for the model.
   *
   * This structure contains configuration parameters used to initialize and
   * manage the model.
   */
  struct common_params params;

  /**
   * @brief Initialization result for the model.
   *
   * This structure holds the result of the model initialization process.
   */
  struct common_init_result llama_init;

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
   * @brief Tokens representing the input prompt.
   *
   * This vector contains the tokenized representation of the input prompt.
   */
  std::vector<llama_token> prompt_tokens;

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

  /**
   * @brief Checks if the prompt contains the prefix at the end.
   *
   * @return True if the prompt contains the prefix, false otherwise.
   */
  bool check_if_prefix();

  /**
   * @brief Load the prefix to the propmt.
   */
  void load_prefix();

  /**
   * @brief Load the suffix to the propmt.
   */
  void load_suffix();

  /**
   * @brief Loads a prompt into the model.
   *
   * @param input_prompt The input text prompt to load.
   * @param add_pfx Whether to add a prefix to the prompt.
   * @param add_sfx Whether to add a suffix to the prompt.
   */
  virtual void load_prompt(const std::string &input_prompt, bool add_pfx,
                           bool add_sfx);

  /**
   * @brief Finds a stopping condition based on the completion results and
   * stopping words.
   *
   * @param completion_result_list A list of completion results to evaluate.
   * @param stopping_words A list of words or phrases that indicate stopping
   * conditions.
   * @return The type of stopping condition encountered.
   */
  StopType
  find_stop(std::vector<struct CompletionOutput> completion_result_list,
            std::vector<std::string> stopping_words);

  /**
   * @brief Finds a stopping condition based on a specific stopping word.
   *
   * @param completion_result_list A list of completion results to evaluate.
   * @param stopping_word A specific word or phrase that indicates a stopping
   * condition.
   * @return The type of stopping condition encountered.
   */
  StopType
  find_stop_word(std::vector<struct CompletionOutput> completion_result_list,
                 std::string stopping_word);

  /**
   * @brief Evaluates the system prompt.
   *
   * @return True if the system prompt evaluation is successful, false
   * otherwise.
   */
  bool eval_system_prompt();

  /**
   * @brief Evaluates the input prompt.
   *
   * This method is virtual and can be overridden by derived classes.
   *
   * @return True if the prompt evaluation is successful, false otherwise.
   */
  virtual bool eval_prompt();

  /**
   * @brief Evaluates a vector of prompt tokens.
   *
   * @param prompt_tokens The vector of tokens to evaluate.
   * @return True if the token evaluation is successful, false otherwise.
   */
  bool eval_prompt(std::vector<llama_token> prompt_tokens);

  /**
   * @brief Evaluates a single token.
   *
   * @param token The token to evaluate.
   * @return True if the token evaluation is successful, false otherwise.
   */
  bool eval_token(llama_token token);

  /**
   * @brief Evaluates a vector of tokens.
   *
   * @param tokens The vector of tokens to evaluate.
   * @return True if the token evaluation is successful, false otherwise.
   */
  bool eval(std::vector<llama_token> tokens);

  /**
   * @brief Evaluates a batch of tokens.
   *
   * This method is virtual and can be overridden by derived classes.
   *
   * @param batch The batch of tokens to evaluate.
   * @return True if the batch evaluation is successful, false otherwise.
   */
  virtual bool eval(struct llama_batch batch);

  /**
   * @brief Retrieves the probabilities of the next tokens.
   *
   * @return A vector of token probabilities.
   */
  std::vector<struct TokenProb> get_probs();

  /**
   * @brief Samples a token based on the current probabilities.
   *
   * @return A structure containing the sampled token and its metadata.
   */
  struct CompletionOutput sample();

private:
  /**
   * @brief A mutex for thread-safe operations.
   */
  std::recursive_mutex mutex;

  /**
   * @brief The last generated chat message while streaming the response.
   * @note slot
   */
  common_chat_msg chat_msg;

  /**
   * @brief A list of generated tool call IDs while streaming the response.
   * @note slot
   */
  std::vector<std::string> generated_tool_call_ids;
};

} // namespace llama_ros

#endif
