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
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include "llama_ros/metadata.hpp"
#include "llama_ros/request_handler.hpp"
#include "llama_ros/result.hpp"
#include "llama_ros/server_slot.hpp"
#include "llama_ros/server_task_result.hpp"
#include "llama_ros/slot_manager.hpp"
#include "llama_ros/task_registry.hpp"
#include "llama_ros/types.hpp"
#include "llama_utils/chat_formatter.hpp"
#include "llama_utils/spinner.hpp"

// Forward declarations to avoid circular dependencies
namespace llama_utils {
struct ChatCompletionsContext;
}

namespace llama_ros {

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
  common_params params;

  /**
   * @brief Constructor for the Llama class.
   *
   * This constructor initializes the Llama object with the given parameters.
   *
   * @param params The common parameters for the model.
   * @param system_prompt The system prompt to be used.
   * @param initial_reset Whether to reset the model initially.
   */
  Llama(const common_params &params, std::string system_prompt = "",
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
  std::vector<LoRA> list_loras();

  /**
   * @brief Updates the current LoRA models with the provided list.
   *
   * @param loras A vector of LoRA structures to update the models.
   */
  void update_loras(std::vector<LoRA> loras);

  /**
   * @brief Generates embeddings for a given input prompt.
   *
   * @param input_prompt The input text prompt for which embeddings are
   * generated.
   * @param normalization The normalization method to apply (default is 2).
   * @return A Result containing embeddings and token count, or an error
   * message.
   */
  Result<ServerTaskResultEmbedding>
  generate_embeddings(const std::string &text);

  void handle_rerank_req(const std::string &query, const std::string &document,
                         ServerSlot *slot);
  void handle_embeddings_req(const std::string &input_prompt, ServerSlot *slot);
  virtual void
  handle_completion_req(const std::string &input_prompt, ServerSlot *slot,
                        common_params_sampling sparams,
                        ServerSlot::GenerateResponseCallback callback,
                        std::vector<std::string> stop, bool reset);
  virtual void
  handle_chat_completion_req(llama_utils::ChatCompletionsContext chat_context,
                             ServerSlot *slot,
                             ServerSlot::GenerateResponseCallback callback);

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
  Result<std::vector<llama_ros::ServerTaskResultRerank>>
  rank_documents(const std::string &query,
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
   * @return A Result containing the generated response and metadata, or an
   * error.
   */
  Result<ServerTaskResultCompletion>
  generate_response(int slot_id, const std::string &input_prompt,
                    common_params_sampling sparams,
                    ServerSlot::GenerateResponseCallback callback = nullptr,
                    std::vector<std::string> stop = {}, bool reset = true);

  Result<ServerTaskResultCompletion> generate_chat_response(
      int slot_gid, llama_utils::ChatCompletionsContext chat_context,
      ServerSlot::GenerateResponseCallback callback = nullptr);

  /**
   * @brief Gets the chat formatter utility.
   *
   * @return Pointer to the chat formatter.
   */
  llama_utils::ChatFormatter *get_chat_formatter() {
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
  common_chat_params get_chat_params(common_chat_templates *tmpls,
                                     common_chat_templates_inputs inputs);

  /**
   * @brief Retrieves performance context data for the model.
   *
   * @return A structure containing performance context data.
   */
  llama_perf_context_data get_perf_data();

  /**
   * @brief Retrieves the internal llama context.
   *
   * @return A pointer to the llama context structure.
   */
  const llama_context *get_ctx() { return this->ctx; }

  /**
   * @brief Retrieves the internal llama model.
   *
   * @return A pointer to the llama model structure.
   */
  const llama_model *get_model() { return this->model; }

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
  const llama_vocab *get_vocab() { return llama_model_get_vocab(this->model); }

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
  Metadata get_metadata();

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
  llama_context *ctx;

  /**
   * @brief Pointer to the llama model.
   *
   * This represents the loaded model used for inference and other operations.
   */
  llama_model *model;

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
  common_sampler *sampler;

  llama_batch batch;

  /**
   * @brief Pointer to the thread pool for parallel processing.
   *
   * This thread pool is used for managing tasks during model execution.
   */
  ggml_threadpool *threadpool;

  /**
   * @brief Pointer to the thread pool for batch processing.
   *
   * This thread pool is used for managing batch operations during model
   * execution.
   */
  ggml_threadpool *threadpool_batch;

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
  std::vector<TokenProb> get_probs(ServerSlot *slot);

  /**
   * @brief Convert raw token probabilities to SelectedLogProb format.
   *
   * @param slot The slot containing generated tokens and probabilities.
   * @return Vector of SelectedLogProb with chosen token and alternatives.
   */
  std::vector<SelectedLogProb> convert_probs_to_logprobs(ServerSlot *slot);

  OAICompactParserOptions oai_parser_opt;
};

} // namespace llama_ros

#endif
