#ifndef LLAMA_HPP
#define LLAMA_HPP

#include <functional>
#include <string>
#include <vector>

#include "llama.h"

struct llama_sampling_params {
  float temp;
  int32_t top_k;
  float top_p;
  float tfs_z;
  float typical_p;
  float repeat_penalty;
  int32_t repeat_last_n;
  float presence_penalty;
  float frequency_penalty;
  int32_t mirostat;
  float mirostat_tau;
  float mirostat_eta;
  bool penalize_nl;
};
struct llama_sampling_params llama_sampling_default_params();

struct llama_eval_params {
  int32_t n_threads;
  int32_t n_predict;
  int32_t n_batch;
  int32_t n_keep;
};
struct llama_eval_params llama_eval_default_params();

namespace llama_ros {

class Llama {

  using GenerateResponseCallback = std::function<void(std::string)>;

public:
  Llama(llama_context_params context_params, llama_eval_params eval_params,
        llama_sampling_params sampling_params, std::string model,
        std::string lora_adapter, std::string lora_base, std::string prefix,
        std::string suffix, std::string stop);
  ~Llama();

  bool embedding;

  std::string detokenize(const std::vector<llama_token> &tokens);
  std::vector<llama_token> tokenize(const std::string &text, bool add_bos);
  void reset();
  void cancel();
  std::vector<float> create_embeddings(const std::string &input_prompt);
  std::string generate_response(const std::string &input_prompt,
                                bool add_pfx_sfx = true,
                                GenerateResponseCallback callbakc = nullptr);

protected:
  llama_context *ctx;
  void eval();
  llama_token sample();

private:
  int32_t n_ctx; // context size

  llama_eval_params eval_params;
  llama_sampling_params sampling_params;

  // prefix, suffix, stop
  std::string stop;
  std::vector<llama_token> inp_pfx;
  std::vector<llama_token> inp_sfx;

  // aux
  std::vector<llama_token> last_n_tokens;
  std::vector<llama_token> prompt_tokens;
  std::vector<llama_token> batch_tokens;

  bool is_antiprompt;
  bool canceled;
  int32_t n_past;
  int32_t n_remain;
  int32_t n_consumed;
};

} // namespace llama_ros

#endif
