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

#include <cassert>
#include <cmath>
#include <thread>

#include "base64.hpp"
#include "common.h"
#include "llama_ros/llava.hpp"

using namespace llama_ros;

Llava::Llava(rclcpp::Logger logger, const struct gpt_params &params, bool debug)
    : logger(logger), params(params), debug(debug), canceled(false) {

  // disable llama.cpp logs
  log_disable();

  if (this->debug) {
    print_build_info();
  }

  // load the model
  this->ctx_llava = this->llava_init(params);
  if (this->ctx_llava == NULL) {
    RCLCPP_ERROR(this->logger, "Failed to init llava");
    return;
  }
}

Llava::~Llava() { this->llava_free(this->ctx_llava); }

struct llava_context *Llava::llava_init(const gpt_params &params) {

  const char *clip_path = params.mmproj.c_str();
  auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/1);

  llama_backend_init();
  llama_numa_init(params.numa);

  llama_model_params model_params = llama_model_params_from_gpt_params(params);
  llama_model *model =
      llama_load_model_from_file(params.model.c_str(), model_params);
  if (model == NULL) {
    RCLCPP_ERROR(this->logger, "Unable to load model");
    return NULL;
  }

  llama_context_params ctx_params =
      llama_context_params_from_gpt_params(params);
  ctx_params.n_ctx = params.n_ctx < 2048
                         ? 2048
                         : params.n_ctx; // we need a longer context size to
                                         // process image embeddings

  llama_context *ctx_llama = llama_new_context_with_model(model, ctx_params);

  if (ctx_llama == NULL) {
    RCLCPP_ERROR(this->logger, "Failed to create the llama_context");
    return NULL;
  }

  auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

  ctx_llava->ctx_llama = ctx_llama;
  ctx_llava->ctx_clip = ctx_clip;
  ctx_llava->model = model;

  return ctx_llava;
}

void Llava::llava_free(struct llava_context *ctx_llava) {

  if (ctx_llava->ctx_clip) {
    clip_free(ctx_llava->ctx_clip);
    ctx_llava->ctx_clip = NULL;
  }

  llama_free(ctx_llava->ctx_llama);
  llama_free_model(ctx_llava->model);
  llama_backend_free();
}

bool Llava::eval_tokens(std::vector<llama_token> tokens, int n_batch,
                        int *n_past) {

  int N = (int)tokens.size();

  for (int i = 0; i < N; i += n_batch) {

    int n_eval = (int)tokens.size() - i;

    if (n_eval > n_batch) {
      n_eval = n_batch;
    }

    if (llama_decode(this->ctx_llava->ctx_llama,
                     llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
      RCLCPP_ERROR(this->logger,
                   "Failed to eval token %d/%d (batch size %d, n_past %d)", i,
                   N, n_batch, *n_past);
      return false;
    }

    *n_past += n_eval;
  }

  return true;
}

bool Llava::eval_id(int id, int *n_past) {
  std::vector<llama_token> tokens;
  tokens.push_back(id);
  return eval_tokens(tokens, 1, n_past);
}

bool Llava::eval_string(const char *str, int n_batch, int *n_past,
                        bool add_bos) {
  std::string str2 = str;
  std::vector<llama_token> embd_inp =
      ::llama_tokenize(this->ctx_llava->ctx_llama, str2, add_bos, true);
  eval_tokens(embd_inp, n_batch, n_past);
  return true;
}

const char *Llava::sample(struct llama_sampling_context *ctx_sampling,
                          int *n_past) {

  const llama_token id =
      llama_sampling_sample(ctx_sampling, this->ctx_llava->ctx_llama, NULL);
  llama_sampling_accept(ctx_sampling, this->ctx_llava->ctx_llama, id, true);

  static std::string ret;

  if (id == llama_token_eos(llama_get_model(this->ctx_llava->ctx_llama))) {
    ret = "</s>";
  } else {
    ret = llama_token_to_piece(this->ctx_llava->ctx_llama, id);
  }

  eval_id(id, n_past);

  return ret.c_str();
}

struct llava_image_embed *Llava::load_image(std::string base64_str) {
  llava_image_embed *embed = this->base64_image_to_embed(base64_str);

  if (!embed) {
    RCLCPP_ERROR(this->logger, "Can't load base64 image");
    return NULL;
  }

  return embed;
}

struct llava_image_embed *
Llava::base64_image_to_embed(const std::string &base64_str) {

  auto required_bytes = base64::required_encode_size(base64_str.size());
  auto img_bytes = std::vector<unsigned char>(required_bytes);
  base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

  auto embed = llava_image_embed_make_with_bytes(
      this->ctx_llava->ctx_clip, this->params.n_threads, img_bytes.data(),
      img_bytes.size());
  if (!embed) {
    RCLCPP_ERROR(this->logger, "Could not load image from base64 string.");
    return NULL;
  }

  return embed;
}

std::string Llava::process_prompt(struct llava_image_embed *image_embed,
                                  const std::string &prompt) {

  int n_past = 0;
  this->canceled = false;

  const int max_tgt_len =
      this->params.n_predict < 0 ? 256 : this->params.n_predict;
  const bool add_bos =
      llama_should_add_bos_token(llama_get_model(this->ctx_llava->ctx_llama));

  std::string system_prompt, user_prompt;
  size_t image_pos = prompt.find("<image>");

  if (image_pos != std::string::npos) {
    system_prompt = prompt.substr(0, image_pos);
    user_prompt = prompt.substr(image_pos + std::string("<image>").length());

  } else {
    // llava-1.5 native mode
    system_prompt =
        "A chat between a curious human and an artificial intelligence "
        "assistant. The assistant gives helpful, detailed, and polite answers "
        "to the human's questions.\nUSER:";
    user_prompt = prompt + "\nASSISTANT:";
  }

  eval_string(system_prompt.c_str(), this->params.n_batch, &n_past, add_bos);
  llava_eval_image_embed(this->ctx_llava->ctx_llama, image_embed,
                         this->params.n_batch, &n_past);
  eval_string(user_prompt.c_str(), this->params.n_batch, &n_past, false);

  // generate the response
  struct llama_sampling_context *ctx_sampling =
      llama_sampling_init(this->params.sparams);

  std::string response = "";
  for (int i = 0; i < max_tgt_len; i++) {

    const char *tmp = sample(ctx_sampling, &n_past);
    response += tmp;

    if (this->canceled) {
      return "";
    }

    if (strcmp(tmp, "</s>") == 0)
      break;
    if (strstr(tmp, "###"))
      break; // Yi-VL behavior
    if (strstr(response.c_str(), "<|im_end|>"))
      break; // Yi-34B llava-1.6 - for some reason those decode not as the
             // correct token (tokenizer works)
    if (strstr(response.c_str(), "<|im_start|>"))
      break; // Yi-34B llava-1.6
    if (strstr(response.c_str(), "USER:"))
      break; // mistral llava-1.6
  }

  llama_sampling_free(ctx_sampling);

  return response;
}

void Llava::cancel() { this->canceled = true; }