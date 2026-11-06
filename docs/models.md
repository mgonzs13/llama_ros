# Predefined Models

[← Back to README](../README.md)

Predefined configuration files live in [`llama_bringup/models`](../llama_bringup/models).
Launch one directly with:

```shell
ros2 llama launch $(ros2 pkg prefix llama_bringup)/share/llama_bringup/models/Llama-3.yaml
```

Or use the generic launch file, selecting `llava_node` for models with a multimodal projector:

```shell
ros2 launch llama_bringup base.launch.py \
  params_file:=$(ros2 pkg prefix llama_bringup)/share/llama_bringup/models/Llama-3.yaml \
  executable:=llama_node
```

| Model YAML                                                              | Hugging Face repo / local path                       | File                                                   | Prompt          | Features                  |
| ----------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ | --------------- | ------------------------- |
| [All-MiniLM-L6-v2](../llama_bringup/models/All-MiniLM-L6-v2.yaml)       | `second-state/All-MiniLM-L6-v2-Embedding-GGUF`       | `all-MiniLM-L6-v2-ggml-model-f16.gguf`                 | —               | embedding                 |
| [Clef](../llama_bringup/models/Clef.yaml)                               | `ggml-org/Clef-Flash-GGUF`                           | `Clef-Flash-Q4_K_M.gguf`                               | —               | multimodal, decision      |
| [DeepSeek-R1](../llama_bringup/models/DeepSeek-R1.yaml)                 | `unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF`           | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf`              | —               | —                         |
| [EmbeddingGemma-2](../llama_bringup/models/EmbeddingGemma-2.yaml)       | `ggml-org/embeddinggemma-2-GGUF`                     | `embeddinggemma-2-Q8_0.gguf`                           | —               | multimodal, embedding     |
| [Falcon3](../llama_bringup/models/Falcon3.yaml)                         | `tiiuae/Falcon3-1B-Instruct-GGUF`                    | `Falcon3-1B-Instruct-q4_k_m.gguf`                      | Phi-3           | —                         |
| [GIST-Embedding](../llama_bringup/models/GIST-Embedding.yaml)           | `mradermacher/GIST-small-Embedding-v0-GGUF`          | `GIST-small-Embedding-v0.Q8_0.gguf`                    | —               | embedding                 |
| [GPT-OSS](../llama_bringup/models/GPT-OSS.yaml)                         | `ggml-org/gpt-oss-20b-GGUF`                          | `gpt-oss-20b-mxfp4.gguf`                               | GPT             | —                         |
| [Gemma-2](../llama_bringup/models/Gemma-2.yaml)                         | `lmstudio-community/gemma-2-2b-it-GGUF`              | `gemma-2-2b-it-Q4_K_M.gguf`                            | Gemma-2         | —                         |
| [Gemma-3](../llama_bringup/models/Gemma-3.yaml)                         | `vinimuchulski/gemma-3-1b-it-qat-q4_0-gguf`          | `gemma-3-1b-it-q4_0.gguf`                              | Gemma-3         | —                         |
| [Gemma-4](../llama_bringup/models/Gemma-4.yaml)                         | `unsloth/gemma-4-E2B-it-GGUF`                        | `gemma-4-E2B-it-Q4_K_M.gguf`                           | Gemma-4         | —                         |
| [Hermes](../llama_bringup/models/Hermes.yaml)                           | `QuantFactory/Hermes-2-Theta-Llama-3-8B-GGUF`        | `Hermes-2-Theta-Llama-3-8B.Q4_K_M.gguf`                | ChatML          | —                         |
| [InternLM](../llama_bringup/models/InternLM.yaml)                       | `internlm/internlm2_5-7b-chat-gguf`                  | `internlm2_5-7b-chat-q4_k_m.gguf`                      | ChatML          | —                         |
| [InternVL3](../llama_bringup/models/InternVL3.yaml)                     | `mradermacher/InternVL3-8B-GGUF`                     | `InternVL3-8B.Q4_K_M.gguf`                             | ChatML          | multimodal                |
| [Julia](../llama_bringup/models/Julia.yaml)                             | `ggml-org/Julia-1-GGUF`                              | `Julia-1-BF16.gguf`                                    | —               | —                         |
| [K2-Horizon](../llama_bringup/models/K2-Horizon.yaml)                   | `bartowski/K2-Horizon-3.7B-GGUF`                     | `K2-Horizon-3.7B-Q4_0.gguf`                            | ChatML          | —                         |
| [Kev](../llama_bringup/models/Kev.yaml)                                 | `ggml-org/Kev-4B-GGUF`                               | `Kev-4B-Q4_K_M.gguf`                                   | —               | decision                  |
| [LFM3](../llama_bringup/models/LFM3.yaml)                               | `LiquidAI/LFM3-1.7B-GGUF`                            | `LFM3-1.7B-Q4_K_M.gguf`                                | ChatML          | —                         |
| [Laya](../llama_bringup/models/Laya.yaml)                               | `ggml-org/Laya-GGUF`                                 | `Laya-Q8_0.gguf`                                       | —               | decision                  |
| [Llama-3-speculative](../llama_bringup/models/Llama-3-speculative.yaml) | `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF` | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`               | Llama-3         | speculative: draft-simple |
| [Llama-3](../llama_bringup/models/Llama-3.yaml)                         | `lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF` | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`               | Llama-3         | —                         |
| [Marcoroni](../llama_bringup/models/Marcoroni.yaml)                     | `TheBloke/Marcoroni-7B-v3-GGUF`                      | `marcoroni-7b-v3.Q4_K_M.gguf`                          | Alpaca          | —                         |
| [MiniCPM-2.5](../llama_bringup/models/MiniCPM-2.5.yaml)                 | `openbmb/MiniCPM-Llama3-V-2_5-gguf`                  | `ggml-model-Q4_K_M.gguf`                               | Llama-3         | multimodal                |
| [MiniCPM-2.6](../llama_bringup/models/MiniCPM-2.6.yaml)                 | `openbmb/MiniCPM-V-2_6-gguf`                         | `ggml-model-Q4_K_M.gguf`                               | ChatML          | multimodal                |
| [MiniCPM-o-2.6](../llama_bringup/models/MiniCPM-o-2.6.yaml)             | `openbmb/MiniCPM-o-2_6-gguf`                         | `Model-7.6B-Q4_K_M.gguf`                               | ChatML          | multimodal                |
| [MiniCPM-v4](../llama_bringup/models/MiniCPM-v4.yaml)                   | `openbmb/MiniCPM-V-4-gguf`                           | `ggml-model-Q4_K_M.gguf`                               | ChatML          | multimodal                |
| [MiniCPM-v5](../llama_bringup/models/MiniCPM-v5.yaml)                   | `mradermacher/MiniCPM5-1B-i1-GGUF`                   | `MiniCPM5-1B.i1-Q4_0.gguf`                             | ChatML          | —                         |
| [Mistral](../llama_bringup/models/Mistral.yaml)                         | `bartowski/Ministral-8B-Instruct-2410-GGUF`          | `Ministral-8B-Instruct-2410-Q4_K_M.gguf`               | Mistral         | —                         |
| [Nimble](../llama_bringup/models/Nimble.yaml)                           | `ggml-org/Bespoke-Nimble-9B-v3-GGUF`                 | `Bespoke-Nimble-9B-v3-Q4_K_M.gguf`                     | —               | decision                  |
| [Phi-3-adapters](../llama_bringup/models/Phi-3-adapters.yaml)           | `bartowski/Phi-3.5-mini-instruct-GGUF`               | `Phi-3.5-mini-instruct-Q4_K_M.gguf`                    | Phi-3           | LoRA                      |
| [Phi-3](../llama_bringup/models/Phi-3.yaml)                             | `bartowski/Phi-3.5-mini-instruct-GGUF`               | `Phi-3.5-mini-instruct-Q4_K_M.gguf`                    | Phi-3           | —                         |
| [Phi-4](../llama_bringup/models/Phi-4.yaml)                             | `bartowski/microsoft_Phi-4-mini-instruct-GGUF`       | `microsoft_Phi-4-mini-instruct-Q4_K_M.gguf`            | Phi-4           | —                         |
| [Qwen2-Audio](../llama_bringup/models/Qwen2-Audio.yaml)                 | `mradermacher/Qwen2-Audio-7B-Instruct-GGUF`          | `Qwen2-Audio-7B-Instruct.Q4_K_M.gguf`                  | ChatML          | multimodal                |
| [Qwen2-VL](../llama_bringup/models/Qwen2-VL.yaml)                       | `bartowski/Qwen2-VL-2B-Instruct-GGUF`                | `Qwen2-VL-2B-Instruct-Q4_K_M.gguf`                     | ChatML          | multimodal                |
| [Qwen2](../llama_bringup/models/Qwen2.yaml)                             | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF`                | `qwen2.5-coder-7b-instruct-q4_k_m-00001-of-00002.gguf` | ChatML          | —                         |
| [Qwen3.5-MTP](../llama_bringup/models/Qwen3.5-MTP.yaml)                 | `unsloth/Qwen3.5-9B-MTP-GGUF`                        | `Qwen3.5-9B-Q4_K_M.gguf`                               | Qwen3           | speculative: draft-mtp    |
| [Qwen3](../llama_bringup/models/Qwen3.yaml)                             | `bartowski/Qwen_Qwen3-0.6B-GGUF`                     | `Qwen_Qwen3-0.6B-Q4_K_M.gguf`                          | Qwen3           | —                         |
| [SmolLM2-slots](../llama_bringup/models/SmolLM2-slots.yaml)             | `bartowski/SmolLM2-1.7B-Instruct-GGUF`               | `SmolLM2-1.7B-Instruct-Q4_K_L.gguf`                    | ChatML          | parallel: 4               |
| [SmolLM2](../llama_bringup/models/SmolLM2.yaml)                         | `bartowski/SmolLM2-1.7B-Instruct-GGUF`               | `SmolLM2-1.7B-Instruct-Q4_K_L.gguf`                    | ChatML          | —                         |
| [Spaetzle](../llama_bringup/models/Spaetzle.yaml)                       | `cstr/Spaetzle-v60-7b-GGUF`                          | `Spaetzle-v60-7b-q4-k-m.gguf`                          | Alpaca          | —                         |
| [StableLM-Zephyr](../llama_bringup/models/StableLM-Zephyr.yaml)         | `duyntnet/stablelm-zephyr-3b-imatrix-GGUF`           | `stablelm-zephyr-3b-Q4_K_M.gguf`                       | StableLM-Zephyr | —                         |
| [Ternary-Bonsai-2](../llama_bringup/models/Ternary-Bonsai-2.yaml)       | `prism-ml/Ternary-Bonsai-2-9B-gguf`                  | `Ternary-Bonsai-2-9B-Q2_0.gguf`                        | ChatML          | —                         |
| [bge-base-en-v1.5](../llama_bringup/models/bge-base-en-v1.5.yaml)       | `CompendiumLabs/bge-base-en-v1.5-gguf`               | `bge-base-en-v1.5-f16.gguf`                            | —               | embedding                 |
| [bge-reranker-v2-m3](../llama_bringup/models/bge-reranker-v2-m3.yaml)   | `gpustack/bge-reranker-v2-m3-GGUF`                   | `bge-reranker-v2-m3-Q4_K_M.gguf`                       | —               | reranking                 |
| [gorilla](../llama_bringup/models/gorilla.yaml)                         | `gorilla-llm/gorilla-openfunctions-v2-gguf`          | `gorilla-openfunctions-v2-q4_K_M.gguf`                 | gorilla         | —                         |
| [harrier-oss](../llama_bringup/models/harrier-oss.yaml)                 | `mykor/harrier-oss-v1-270m-GGUF`                     | `harrier-oss-v1-270M-Q8_0.gguf`                        | —               | embedding                 |
| [jina-reranker](../llama_bringup/models/jina-reranker.yaml)             | `gpustack/jina-reranker-v1-tiny-en-GGUF`             | `jina-reranker-v1-tiny-en-FP16.gguf`                   | —               | reranking                 |
| [llava-mistral](../llama_bringup/models/llava-mistral.yaml)             | `cjpais/llava-1.6-mistral-7b-gguf`                   | `llava-v1.6-mistral-7b.Q4_K_M.gguf`                    | Mistral         | multimodal                |
| [llava-phi-3](../llama_bringup/models/llava-phi-3.yaml)                 | `xtuner/llava-phi-3-mini-gguf`                       | `llava-phi-3-mini-int4.gguf`                           | Phi-3           | multimodal                |
| [localmentor](../llama_bringup/models/localmentor.yaml)                 | `mgonzs13/stablelm-zephyr-3B-localmentor-GGUF`       | `stablelm-zephyr-3B-localmentor.Q4_K_M.gguf`           | Zephyr          | —                         |
| [multilingual-e5](../llama_bringup/models/multilingual-e5.yaml)         | `nnch/multilingual-e5-large-Q4_K_M-GGUF`             | `multilingual-e5-large-q4_k_m.gguf`                    | —               | embedding                 |
| [mxbai-embed](../llama_bringup/models/mxbai-embed.yaml)                 | `mixedbread-ai/mxbai-embed-large-v1`                 | `gguf/mxbai-embed-large-v1-f16.gguf`                   | —               | embedding                 |
| [pplx-decider](../llama_bringup/models/pplx-decider.yaml)               | `Jakevin/pplx-decider-flash-v1.2-GGUF`               | `pplx-decider-flash-v1.2-Q4_K_M.gguf`                  | —               | decision                  |
| [vllm-d3](../llama_bringup/models/vllm-d3.yaml)                         | `ggml-org/vllm-d3-nano-GGUF`                         | `vllm-d3-nano-Q8_0.gguf`                               | —               | multimodal, decision      |
