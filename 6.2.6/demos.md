# Demos

[← Back to README](../README.md)

## LLM Demo

```shell
ros2 launch llama_bringup spaetzle.launch.py
```

```shell
ros2 run llama_demos llama_demo_node
```

<!-- https://user-images.githubusercontent.com/25979134/229344687-9dda3446-9f1f-40ab-9723-9929597a042c.mp4 -->

https://github.com/mgonzs13/llama_ros/assets/25979134/9311761b-d900-4e58-b9f8-11c8efefdac4

## Speculative Decoding Demo

```shell
ros2 launch llama_bringup llama-3-speculative.launch.py
```

```shell
ros2 run llama_demos llama_demo_node
```

## MTP Speculative Decoding Demo

```shell
ros2 launch llama_bringup Qwen3.5-MTP.launch.py
```

```shell
ros2 run llama_demos chatllama_demo_node
```

## Embeddings Generation Demo

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/bge-base-en-v1.5.yaml
```

```shell
ros2 run llama_demos llama_embeddings_demo_node
```

https://github.com/user-attachments/assets/7d722017-27dc-417c-ace7-bf6b747e4ced

## Multimodal Embeddings Generation Demo

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/EmbeddingGemma-2.yaml
```

```shell
ros2 run llama_demos llama_multimodal_embeddings_demo_node
```

## Reranking Demo

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/jina-reranker.yaml
```

```shell
ros2 run llama_demos llama_rerank_demo_node
```

https://github.com/user-attachments/assets/4b4adb4d-7c70-43ea-a2c1-9be57d211484

## Decision Demo

`llama_ros` supports llama.cpp decision models: OpenJEV, Lev, Kev, Nimble,
Laya and Clef. A decision model scores the options of typed questions about a
state without generating text. The type is detected from the model metadata,
so any upstream decision GGUF works.

The node exposes an `evaluate_decisions` service
(`llama_msgs/srv/EvaluateDecisions`):

```
string state                 # one state shared by all questions (JSON or text)
sensor_msgs/Image[] images   # optional; OpenJEV and Clef
DecisionQuestion[] questions # must be non-empty
---
DecisionAnswer[] answers     # aligned 1:1 with questions
```

Each question is a `choice` (pick one option key), a `score` (rate along 2 to
10 ordered levels) or a `noul` (probability that a statement holds). Answers
carry `success`/`error`; choice and score answers also carry probabilities for
the option keys and a `confidence` (a `noul` answer reports its probability in
`noul`, and `confidence` stays 0). A failing question does not prevent the
other questions from being answered; Clef is the exception, since it answers
all questions in one joint pass and fails them together. Images are accepted
for OpenJEV, Clef, Pplx-decider and Vllm-d3, up to 8 per request, and require
the `llava` node with a multimodal projector.

Supported models and their GGUFs: `ggml-org/OpenJEV-GGUF`,
`ggml-org/Laya-GGUF`, `ggml-org/Kev-4B-GGUF` (also 0.8B/9B),
`ggml-org/Bespoke-Nimble-9B-v3-GGUF`, `ggml-org/Clef-GGUF` (also Flash) and
`ggml-org/Julia-1-GGUF`.

Run the demo:

```shell
ros2 launch llama_bringup decision.launch.py
```

```shell
ros2 run llama_demos decision_demo_node
```

The demo sends all questions of a scenario in one batch request. Note that
questions are currently decoded independently (no shared-prompt-prefix
grouping yet), except Clef which answers all questions of a request in one
joint forward pass.

## Image Decision Demo

Clef, OpenJEV, Pplx-decider and Vllm-d3 decision models accept images through
the `llava` node with a multimodal projector. The image decision demo downloads
a fixed image from a URL (pass a different URL as the first argument to
override it) and sends it with a `choice`, a `score` and a `noul` question;
Clef answers all of them in one joint forward pass, the other types evaluate
one question per pass.

```shell
ros2 launch llama_bringup image-decision.launch.py
```

The launch file uses Clef; other image-capable models work too, for example:

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/vllm-d3.yaml
```

```shell
ros2 run llama_demos image_decision_demo_node [image_url]
```

Up to 8 images are accepted per request. The whole prompt (text plus image
embeddings) must fit in one batch, so `context.n_batch`/`n_ubatch` must be
large enough; the multimodal projector's `image_max_tokens` is capped to half
the ubatch for joint decision models.

## RAG Demo (LLM + chat template + RAG + Reranking + Stream)

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/bge-base-en-v1.5.yaml
```

```shell
ros2 llama launch ~/ros2_ws/src/llama_ros/llama_bringup/models/jina-reranker.yaml
```

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    context:
      n_ctx: 4096
      n_batch: 256
      n_ubatch: 256
      n_predict: -1
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
    prompt:
      stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos llama_rag_demo_node
```

https://github.com/user-attachments/assets/b4e3957d-1f92-427b-a1a8-cfc76737c0d6

## Chat Template Demo

```shell
ros2 llama launch MiniCPM-2.6.yaml
```

<details>
<summary>Click to expand MiniCPM-2.6.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "ggml-model-Q4_K_M.gguf"
    mmproj:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "mmproj-model-f16.gguf"
    context:
      n_ctx: 8192
      n_batch: 512
      n_predict: 8192
    gpu:
      n_gpu_layers: 20
    cpu:
      n_threads: -1
```

</details>

```shell
ros2 run llama_demos chatllama_image_demo_node
```

[ChatLlamaROS demo](https://github-production-user-asset-6210df.s3.amazonaws.com/55236157/363094669-c6de124a-4e91-4479-99b6-685fecb0ac20.webm?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240830%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240830T081232Z&X-Amz-Expires=300&X-Amz-Signature=f937758f4bcbaec7683e46ddb057fb642dc86a33cc8c736fca3b5ce2bf06ddac&X-Amz-SignedHeaders=host&actor_id=55236157&key_id=0&repo_id=622137360)

## VLM Demo

```shell
ros2 launch llama_bringup minicpm-2.6.launch.py
```

```shell
ros2 run llama_demos llava_demo_node
```

https://github.com/mgonzs13/llama_ros/assets/25979134/4a9ef92f-9099-41b4-8350-765336e3503c

## Chat Multi-Image Demo

```shell
ros2 llama launch MiniCPM-2.6.yaml
```

<details>
<summary>Click to expand MiniCPM-2.6.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "ggml-model-Q4_K_M.gguf"
    mmproj:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "mmproj-model-f16.gguf"
    context:
      n_ctx: 8192
      n_batch: 512
      n_predict: 8192
    gpu:
      n_gpu_layers: 20
    cpu:
      n_threads: -1
```

</details>

```shell
ros2 run llama_demos chatllama_multi_image_demo_node
```

## Chat Multi-Image (User Input) Demo

```shell
ros2 llama launch MiniCPM-2.6.yaml
```

<details>
<summary>Click to expand MiniCPM-2.6.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "ggml-model-Q4_K_M.gguf"
    mmproj:
      repo: "openbmb/MiniCPM-V-2_6-gguf"
      filename: "mmproj-model-f16.gguf"
    context:
      n_ctx: 8192
      n_batch: 512
      n_predict: 8192
    gpu:
      n_gpu_layers: 20
    cpu:
      n_threads: -1
```

</details>

```shell
ros2 run llama_demos chatllama_multi_image_user_demo_node
```

## MTMD Audio Demo

```shell
ros2 llama launch Qwen2-Audio.yaml
```

<details>
<summary>Click to expand Qwen2-Audio.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.Q4_K_M.gguf
    mmproj:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.mmproj-f16.gguf
    context:
      n_ctx: 8192
      n_batch: 512
      n_predict: 8192
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
    prompt:
      system_prompt_type: ChatML
```

</details>

```shell
ros2 run llama_demos mtmd_audio_demo_node
```

## Chat Audio Demo

```shell
ros2 llama launch Qwen2-Audio.yaml
```

<details>
<summary>Click to expand Qwen2-Audio.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.Q4_K_M.gguf
    mmproj:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.mmproj-f16.gguf
    context:
      n_ctx: 8192
      n_batch: 512
      n_predict: 8192
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
```

</details>

```shell
ros2 run llama_demos chatllama_audio_demo_node
```

## Chat Multi-Audio Demo

```shell
ros2 llama launch Qwen2-Audio.yaml
```

<details>
<summary>Click to expand Qwen2-Audio.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.Q4_K_M.gguf
    mmproj:
      repo: mradermacher/Qwen2-Audio-7B-Instruct-GGUF
      filename: Qwen2-Audio-7B-Instruct.mmproj-f16.gguf
    context:
      n_ctx: 8192
      n_batch: 512
      n_predict: 8192
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
```

</details>

```shell
ros2 run llama_demos chatllama_multi_audio_demo_node
```

## Chat Structured Output Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    context:
      n_ctx: 4096
      n_batch: 256
      n_ubatch: 256
      n_predict: -1
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
    prompt:
      stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_structured_demo_node
```

[Structured Output ChatLlama](https://github.com/user-attachments/assets/e0bf4031-50c0-4790-94a0-1f6aed5734ec)

## Chat Tools Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    context:
      n_ctx: 4096
      n_batch: 256
      n_ubatch: 256
      n_predict: -1
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
    prompt:
      stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_tools_demo_node
```

[Tools ChatLlama](https://github.com/user-attachments/assets/b912ee29-1466-4d6a-888b-9a2d9c16ae1d)

## Streaming Tools Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    context:
      n_ctx: 4096
      n_batch: 256
      n_ubatch: 256
      n_predict: -1
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
    prompt:
      stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_streaming_tools_demo_node
```

## Chat Reasoning Demo (DeepSeek-R1)

```shell
ros2 llama launch DeepSeek-R1.yaml
```

<details>
<summary>Click to expand DeepSeek-R1.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF
      filename: DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
    context:
      n_ctx: 4096
      n_batch: 256
      n_ubatch: 256
      n_predict: -1
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: 1
    prompt:
      chat_template_file: llama-cpp-deepseek-r1.jinja
```

</details>

```shell
ros2 run llama_demos chatllama_reasoning_demo_node
```

[DeepSeekR1 ChatLlama](https://github.com/user-attachments/assets/3f268614-eabc-4499-b50f-a76d76908d9d)

## Reasoning + Tools Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    context:
      n_ctx: 4096
      n_batch: 256
      n_ubatch: 256
      n_predict: -1
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
    prompt:
      stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_reasoning_tools_demo_node
```

## PDDL Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    context:
      n_ctx: 4096
      n_batch: 256
      n_ubatch: 256
      n_predict: -1
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
    prompt:
      stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_pddl_demo_node
```

## Agent Demo

```shell
ros2 llama launch Qwen3.yaml
```

<details>
<summary>Click to expand Qwen3.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: bartowski/Qwen_Qwen3-8B-GGUF
      filename: Qwen_Qwen3-8B-Q4_K_M.gguf
    context:
      n_ctx: 4096
      n_batch: 256
      n_ubatch: 256
      n_predict: -1
    gpu:
      n_gpu_layers: -1
    cpu:
      n_threads: -1
    prompt:
      stopping_words: ["<|im_end|>"]
```

</details>

```shell
ros2 run llama_demos chatllama_agent_demo_node
```

[Agent ChatLlama](https://github.com/user-attachments/assets/a0991cb4-f7f4-43d5-b629-3b1819aead0d)

## Parallel Slots Demo

This demo shows how to use multiple parallel slots (`context.n_parallel`) to process several requests concurrently via continuous batching. Launch the model with `n_parallel: 4`:

```shell
ros2 llama launch SmolLM2-slots.yaml
```

<details>
<summary>Click to expand SmolLM2-slots.yaml</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: bartowski/SmolLM2-1.7B-Instruct-GGUF
      filename: SmolLM2-1.7B-Instruct-Q4_K_L.gguf
    context:
      n_ctx: 2048
      n_batch: 8
      n_ubatch: 8
      n_predict: 2048
      n_parallel: 4
    gpu:
      n_gpu_layers: 0
    cpu:
      n_threads: -1
    prompt:
      system_prompt_type: ChatML
```

</details>

```shell
ros2 run llama_demos llama_slots_demo_node
```

## Evaluate a prompt without generating text

Set `GenerateResponse.Goal.precompute` to `true` to evaluate a text prompt and
retain its computed KV state without sampling output tokens. A successful action
returns empty text and an empty token list. The default is `false`, preserving
normal generation. Evaluation uses the configured model CPU and batch settings;
there are no separate per-request resource budgets.

Preparation currently requires one slot, prompt caching enabled, a text-only
request with `reset=false`, and a non-recurrent, non-hybrid model without sliding-window attention
or speculative decoding. Unsupported or busy preparation requests are rejected.
Later requests can reuse compatible cached tokens through the existing cache
machinery. Always send the complete current prompt, not just its new suffix.

Use the existing ROS action cancellation protocol to interrupt a request. The
caller must wait for its terminal result before relying on the slot being free.
Request priority belongs to the caller: submitting another request does not
implicitly cancel an active preparation.

For supported single-slot text contexts, CPU evaluation can stop cooperatively
inside `llama_decode`; completed microbatches are retained and unfinished work
is discarded. This does not interrupt an individual running operation. Other
configurations cancel at processing boundaries. Cache reuse after cancellation
is conditional on the model, retained state and the next prompt; cancellation
does not promise a fixed latency on every backend.

The CLI uses the same action client:

```shell
ros2 llama prompt "<complete prompt>" --precompute
```

`--action-name` selects the endpoint (default `/llama/generate_response`).
SIGINT/SIGTERM request cancellation. Subprocess callers can pass an inherited
pipe read descriptor with `--cancel-fd N`; closing its write end cancels that
request, including while goal acceptance is pending. The CLI waits up to five
seconds for cancellation to finish. Successful preparation prints no text and
exits zero; cancellation, rejection and errors exit nonzero.

This adds a field to the ROS action wire definition. Rebuild `llama_msgs` and
all clients and dependents together before using it.
