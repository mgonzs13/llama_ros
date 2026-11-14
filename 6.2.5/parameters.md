# ROS 2 Parameters

[← Back to README](../README.md)

The following tables list all the ROS 2 parameters available when launching `llama_node` or `llava_node`. Parameters are organized by namespace.

## General

| Param       | Type    | Default | Description                                 |
| ----------- | ------- | ------- | ------------------------------------------- |
| `verbosity` | `int32` | `3`     | Log verbosity level                         |
| `no_alloc`  | `bool`  | `false` | Disable memory allocation for output tokens |

## Model (`model.*`)

| Param                 | Type       | Default | Description                                           |
| --------------------- | ---------- | ------- | ----------------------------------------------------- |
| `model.path`          | `string`   | `""`    | Local file path to the GGUF model                     |
| `model.repo`          | `string`   | `""`    | HuggingFace repository ID to download the model from  |
| `model.filename`      | `string`   | `""`    | Filename of the model in the HuggingFace repository   |
| `model.warmup`        | `bool`     | `true`  | Run a warmup inference on load                        |
| `model.check_tensors` | `bool`     | `false` | Validate model tensor data on load                    |
| `model.load_mtp`      | `bool`     | `false` | Load MTP/NextN layers for multi-token prediction      |
| `model.shards`        | `string[]` | `[]`    | Extra GGUF shards to load in addition to `model.path` |

## Multimodal Projector (`mmproj.*`)

| Param             | Type     | Default | Description                                                                |
| ----------------- | -------- | ------- | -------------------------------------------------------------------------- |
| `mmproj.path`     | `string` | `""`    | Local file path to the multimodal projector                                |
| `mmproj.repo`     | `string` | `""`    | HuggingFace repository ID to download the projector from                   |
| `mmproj.filename` | `string` | `""`    | Filename of the projector in the HuggingFace repository                    |
| `mmproj.use_gpu`  | `bool`   | `true`  | Use GPU for the multimodal projector                                       |
| `mmproj.device`   | `string` | `""`    | Device for the projector (`none` disables GPU, `""` follows `gpu.devices`) |
| `mmproj.disabled` | `bool`   | `false` | Disable loading the multimodal projector                                   |

## Context / Inference (`context.*`)

| Param                           | Type     | Default  | Description                                                                     |
| ------------------------------- | -------- | -------- | ------------------------------------------------------------------------------- |
| `context.seed`                  | `int32`  | `-1`     | RNG seed for sampling (`-1` for default)                                        |
| `context.n_ctx`                 | `int32`  | `0`      | Context size in tokens (`0` for model default)                                  |
| `context.n_batch`               | `int32`  | `2048`   | Logical batch size for prompt processing                                        |
| `context.n_ubatch`              | `int32`  | `512`    | Physical batch size                                                             |
| `context.n_keep`                | `int32`  | `0`      | Number of tokens to keep from the initial prompt on context shift               |
| `context.n_chunks`              | `int32`  | `-1`     | Max number of chunks to process (`-1` for unlimited)                            |
| `context.n_predict`             | `int32`  | `-1`     | Max tokens to predict (`-1` for unlimited when using ctx_shift)                 |
| `context.n_parallel`            | `int32`  | `1`      | Number of parallel sequences to decode                                          |
| `context.n_outputs_max`         | `int32`  | `0`      | Maximum number of outputs in a batch (`0` for `n_batch`)                        |
| `context.n_sequences`           | `int32`  | `1`      | Number of sequences to decode                                                   |
| `context.n_outputs_max_per_seq` | `int32`  | `1`      | Maximum number of outputs per sequence                                          |
| `context.numa`                  | `string` | `"none"` | NUMA strategy: `none`, `distribute`, `isolate`, `numactl`, `mirror`, or `count` |
| `context.pooling_type`          | `string` | `""`     | Pooling type: `none`, `mean`, `cls`, `last`, or `rerank`                        |
| `context.attention_type`        | `string` | `""`     | Attention type: `causal` or `non_causal`                                        |
| `context.embd_out`              | `string` | `""`     | Embedding output format: `""` (default), `array`, `json`, or `json+`            |
| `context.embd_sep`              | `string` | `"\n"`   | Separator between embeddings                                                    |
| `context.cls_sep`               | `string` | `"\t"`   | Separator between classification sequences                                      |
| `context.embedding`             | `bool`   | `false`  | Enable embedding mode                                                           |
| `context.reranking`             | `bool`   | `false`  | Enable reranking mode (sets pooling to `rerank` and enables embedding)          |
| `context.ctx_shift`             | `bool`   | `false`  | Enable context shifting                                                         |
| `context.swa_full`              | `bool`   | `false`  | Enable full sliding window attention                                            |
| `context.cont_batching`         | `bool`   | `true`   | Enable continuous batching                                                      |

## GPU / Backend (`gpu.*`)

| Param                   | Type       | Default        | Description                                                                   |
| ----------------------- | ---------- | -------------- | ----------------------------------------------------------------------------- |
| `gpu.n_gpu_layers`      | `int32`    | `-1`           | Number of layers to offload to GPU (`-1` for all)                             |
| `gpu.main_gpu`          | `int32`    | `0`            | Main GPU index                                                                |
| `gpu.split_mode`        | `string`   | `"layer"`      | GPU split mode: `none`, `layer`, or `row`                                     |
| `gpu.flash_attn_type`   | `string`   | `"auto"`       | Flash attention type: `auto`, `enabled`, or `disabled`                        |
| `gpu.tensor_split`      | `double[]` | `[0.0]`        | Tensor split proportions across GPUs                                          |
| `gpu.devices`           | `string[]` | `[]`           | GPU device names to use                                                       |
| `gpu.fit_params_target` | `int64[]`  | `[1073741824]` | Per-device memory target in bytes for automatic VRAM fitting (one per device) |
| `gpu.no_kv_offload`     | `bool`     | `false`        | Disable KV cache offloading to GPU                                            |
| `gpu.no_op_offload`     | `bool`     | `false`        | Disable operation offloading                                                  |
| `gpu.no_host`           | `bool`     | `false`        | Disable host buffer usage                                                     |
| `gpu.no_extra_bufts`    | `bool`     | `false`        | Disable extra buffer types                                                    |

## Model Overrides

| Param                   | Type       | Default | Description                                                                                                                        |
| ----------------------- | ---------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `kv_overrides`          | `string[]` | `[]`    | Override GGUF model metadata key-value pairs. Format: `"key=int:123"`, `"key=float:3.14"`, `"key=bool:true"`, or `"key=str:value"` |
| `tensor_buft_overrides` | `string[]` | `[]`    | Override tensor buffer types by regex pattern. Format: `"pattern=CPU"` (buft name from available backends)                         |

## Control Vector (`control_vector.*`)

| Param                        | Type       | Default | Description                                                                                   |
| ---------------------------- | ---------- | ------- | --------------------------------------------------------------------------------------------- |
| `control_vector.layer_start` | `int32`    | `-1`    | Start layer for control vector application                                                    |
| `control_vector.layer_end`   | `int32`    | `-1`    | End layer for control vector application                                                      |
| `control_vectors`            | `string[]` | `[]`    | Control vector files with optional strength. Format: `"path/to/cvector.gguf"` or `"path@0.5"` |

## Multimodal (`multimodal.*`)

| Param                              | Type    | Default | Description                                        |
| ---------------------------------- | ------- | ------- | -------------------------------------------------- |
| `multimodal.image_min_tokens`      | `int32` | `-1`    | Minimum number of tokens per image (`-1` for auto) |
| `multimodal.image_max_tokens`      | `int32` | `-1`    | Maximum number of tokens per image (`-1` for auto) |
| `multimodal.mtmd_batch_max_tokens` | `int32` | `1024`  | Maximum batch tokens for multimodal data           |

## Memory (`memory.*`)

| Param                     | Type     | Default  | Description                                                 |
| ------------------------- | -------- | -------- | ----------------------------------------------------------- |
| `memory.load_mode`        | `string` | `"auto"` | Mode to load the model (auto, none, mmap, mlock, direct_io) |
| `memory.lazy_mode`        | `string` | `"auto"` | On-demand tensor reading (off, auto, on) (requires mmap)    |
| `memory.kv_unified`       | `bool`   | `false`  | Use unified KV cache                                        |
| `memory.cache_idle_slots` | `bool`   | `true`   | Save and clear idle KV cache slots when a new task starts   |

## CPU (`cpu.*`)

| Param           | Type     | Default    | Description                                                                  |
| --------------- | -------- | ---------- | ---------------------------------------------------------------------------- |
| `cpu.n_threads` | `int32`  | `-1`       | Number of threads for generation (`-1` for auto-detect)                      |
| `cpu.poll`      | `int32`  | `50`       | Thread pool polling interval                                                 |
| `cpu.mask`      | `string` | `""`       | CPU affinity mask for generation threads                                     |
| `cpu.range`     | `string` | `""`       | CPU range for generation threads                                             |
| `cpu.priority`  | `string` | `"normal"` | Thread scheduling priority: `low`, `normal`, `medium`, `high`, or `realtime` |
| `cpu.strict`    | `bool`   | `false`    | Strict CPU affinity for generation threads                                   |

## CPU Batch (`cpu_batch.*`)

| Param                 | Type     | Default    | Description                                                   |
| --------------------- | -------- | ---------- | ------------------------------------------------------------- |
| `cpu_batch.n_threads` | `int32`  | `-1`       | Number of threads for batch processing (`-1` for auto-detect) |
| `cpu_batch.poll`      | `int32`  | `50`       | Thread pool polling interval for batch processing             |
| `cpu_batch.mask`      | `string` | `""`       | CPU affinity mask for batch processing threads                |
| `cpu_batch.range`     | `string` | `""`       | CPU range for batch processing threads                        |
| `cpu_batch.priority`  | `string` | `"normal"` | Thread scheduling priority for batch processing               |
| `cpu_batch.strict`    | `bool`   | `false`    | Strict CPU affinity for batch processing threads              |

## RoPE (`rope.*`)

| Param               | Type     | Default | Description                                                |
| ------------------- | -------- | ------- | ---------------------------------------------------------- |
| `rope.freq_base`    | `float`  | `0.0`   | RoPE base frequency (`0.0` for model default)              |
| `rope.freq_scale`   | `float`  | `0.0`   | RoPE frequency scale factor (`0.0` for model default)      |
| `rope.scaling_type` | `string` | `""`    | RoPE scaling type: `none`, `linear`, `yarn`, or `longrope` |

## YaRN (`yarn.*`)

| Param              | Type    | Default | Description                                              |
| ------------------ | ------- | ------- | -------------------------------------------------------- |
| `yarn.ext_factor`  | `float` | `-1.0`  | YaRN extrapolation mix factor (`-1.0` for model default) |
| `yarn.attn_factor` | `float` | `-1.0`  | YaRN attention magnitude scaling factor                  |
| `yarn.beta_fast`   | `float` | `-1.0`  | YaRN low correction dimension                            |
| `yarn.beta_slow`   | `float` | `-1.0`  | YaRN high correction dimension                           |
| `yarn.orig_ctx`    | `int32` | `0`     | YaRN original context size                               |

## Group Attention (`grp_attn.*`)

| Param        | Type    | Default | Description                                           |
| ------------ | ------- | ------- | ----------------------------------------------------- |
| `grp_attn.n` | `int32` | `1`     | Self-extend group attention factor (`1` for disabled) |
| `grp_attn.w` | `int32` | `512`   | Self-extend group attention width                     |

## KV Cache (`cache.*`)

| Param                       | Type     | Default | Description                                                                                                                      |
| --------------------------- | -------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `cache.type_k`              | `string` | `"f16"` | Data type for K cache: `f32`, `f16`, `bf16`, `f8_e4m3`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, or `q5_1`                      |
| `cache.type_v`              | `string` | `"f16"` | Data type for V cache (same options as `cache.type_k`)                                                                           |
| `cache.cache_prompt`        | `bool`   | `true`  | Enable prompt caching (reuse previously evaluated KV state)                                                                      |
| `cache.cache_ram_mib`       | `int32`  | `8192`  | RAM limit for the prompt cache in MiB (`-1` = no limit, `0` = disable cache)                                                     |
| `cache.cache_ram_explicit`  | `bool`   | `false` | Mark the prompt cache as explicitly configured. Decision models disable the prompt cache unless this is `true`                   |
| `cache.moe_cache_mib`       | `int32`  | `0`     | GPU cache size in MiB for the MoE experts kept in host memory (split among GPUs like the layers; `0` = disabled)                 |
| `cache.moe_cache_cpu_share` | `float`  | `0.0`   | Share of MoE cache misses computed on the CPU instead of being uploaded to the GPU, in `[0.0, 1.0]` (`0.0` = upload all)         |
| `cache.n_cache_reuse`       | `int32`  | `0`     | Minimum chunk size in tokens to reuse from the KV cache via shifting (`0` = disabled)                                            |
| `cache.n_ctx_checkpoints`   | `int32`  | `32`    | Maximum number of context checkpoints per slot (`0` = disabled)                                                                  |
| `cache.checkpoint_min_step` | `int32`  | `256`   | Minimum spacing between context checkpoints                                                                                      |
| `cache.kv_unified_per_slot` | `int32`  | `0`     | Context limit per parallel slot (`0` = unset). When set with `context.n_ctx: 0`, the shared KV pool is sized to `n_parallel * N` |

## Fit Parameters (`fit.*`)

| Param         | Type    | Default | Description                                            |
| ------------- | ------- | ------- | ------------------------------------------------------ |
| `fit.enabled` | `bool`  | `true`  | Automatically fit model parameters to available memory |
| `fit.min_ctx` | `int32` | `4096`  | Minimum context size when fitting parameters           |

## Speculative Decoding (`speculative.*`)

Speculative decoding uses draft tokens to accelerate generation, then verifies them in parallel with the main model. llama_ros supports both draft-model-based and self-speculative (ngram-based) approaches.

**Note:** Speculative decoding requires `context.n_parallel: 1` (single slot) and is not supported with embedding/reranking models.

| Param                     | Type       | Default  | Description                                                                                                                                                                                   |
| ------------------------- | ---------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `speculative.type`        | `string`   | `"none"` | Speculative decoding type: `none`, `draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, or `ngram-cache` |
| `speculative.synth_len`   | `double`   | `-1.0`   | Target mean synthetic acceptance length, including the target token (benchmarking only, `-1.0` = disabled)                                                                                    |
| `speculative.synth_rates` | `double[]` | `[]`     | Unconditional per-position synthetic acceptance probabilities (benchmarking only)                                                                                                             |

### Draft Model (`speculative.draft.*`)

Used with draft-model-based types (`draft-simple`, `draft-eagle3`, `draft-mtp`, `draft-dflash`, `draft-dspark`).

| Param                                     | Type       | Default | Description                                                                                                                                                                                             |
| ----------------------------------------- | ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `speculative.draft.n_max`                 | `int32`    | `16`    | Maximum number of tokens to draft per speculative step                                                                                                                                                  |
| `speculative.draft.n_min`                 | `int32`    | `0`     | Minimum number of draft tokens required to attempt verification. If the draft model produces fewer tokens than this, the draft is discarded and a single token is generated instead. `0` is recommended |
| `speculative.draft.p_min`                 | `double`   | `0.75`  | Minimum probability threshold for draft tokens (greedy)                                                                                                                                                 |
| `speculative.draft.p_split`               | `double`   | `0.1`   | Split probability threshold for speculative sampling                                                                                                                                                    |
| `speculative.draft.n_gpu_layers`          | `int32`    | `-1`    | Number of layers to offload to GPU for the draft model (`-1` for all)                                                                                                                                   |
| `speculative.draft.n_ubatch`              | `int32`    | `0`     | Physical batch size of the draft context; use a smaller value to spare VRAM (`0` = use the target's `context.n_ubatch`)                                                                                 |
| `speculative.draft.cache_type_k`          | `string`   | `"f16"` | KV cache type for K in the draft model (e.g. `f16`, `q8_0`, `q4_0`)                                                                                                                                     |
| `speculative.draft.cache_type_v`          | `string`   | `"f16"` | KV cache type for V in the draft model (e.g. `f16`, `q8_0`, `q4_0`)                                                                                                                                     |
| `speculative.draft.model.path`            | `string`   | `""`    | Local file path to the draft model GGUF file                                                                                                                                                            |
| `speculative.draft.model.repo`            | `string`   | `""`    | HuggingFace repository ID for the draft model                                                                                                                                                           |
| `speculative.draft.model.filename`        | `string`   | `""`    | Filename of the draft model in the HuggingFace repository                                                                                                                                               |
| `speculative.draft.backend_sampling`      | `bool`     | `true`  | Whether to offload draft sampling to the backend                                                                                                                                                        |
| `speculative.draft.probabilistic`         | `bool`     | `false` | Sample the draft and verify by rejection instead of argmax and match                                                                                                                                    |
| `speculative.draft.devices`               | `string[]` | `[]`    | GPU devices to use for the draft model (device names, e.g. `CUDA0`)                                                                                                                                     |
| `speculative.draft.tensor_buft_overrides` | `string[]` | `[]`    | Draft tensor buffer overrides, format `pattern=buft_name`                                                                                                                                               |

### Draft CPU (`speculative.draft.cpu.*` and `speculative.draft.cpu_batch.*`)

Thread configuration for the draft model, same fields as `cpu.*` / `cpu_batch.*`.

| Param                                   | Type     | Default    | Description                                                             |
| --------------------------------------- | -------- | ---------- | ----------------------------------------------------------------------- |
| `speculative.draft.cpu.n_threads`       | `int32`  | `-1`       | Draft CPU threads (`-1` for auto)                                       |
| `speculative.draft.cpu.mask`            | `string` | `""`       | Draft CPU affinity mask                                                 |
| `speculative.draft.cpu.range`           | `string` | `""`       | Draft CPU affinity range                                                |
| `speculative.draft.cpu.priority`        | `string` | `"normal"` | Draft thread priority: `low`, `normal`, `medium`, `high`, or `realtime` |
| `speculative.draft.cpu.strict`          | `bool`   | `false`    | Fail if the draft CPU affinity cannot be set                            |
| `speculative.draft.cpu.poll`            | `int32`  | `50`       | Draft polling level (`0`-`100`)                                         |
| `speculative.draft.cpu_batch.n_threads` | `int32`  | `-1`       | Draft CPU batch threads (`-1` for auto)                                 |
| `speculative.draft.cpu_batch.mask`      | `string` | `""`       | Draft CPU batch affinity mask                                           |
| `speculative.draft.cpu_batch.range`     | `string` | `""`       | Draft CPU batch affinity range                                          |
| `speculative.draft.cpu_batch.priority`  | `string` | `"normal"` | Draft batch thread priority                                             |
| `speculative.draft.cpu_batch.strict`    | `bool`   | `false`    | Fail if the draft batch affinity cannot be set                          |
| `speculative.draft.cpu_batch.poll`      | `int32`  | `50`       | Draft batch polling level (`0`-`100`)                                   |

### Ngram-mod (`speculative.ngram_mod.*`)

Used with `speculative.type: ngram-mod`. A hash-based self-speculative method that learns n-gram patterns from the prompt without a separate model.

| Param                           | Type    | Default | Description                                           |
| ------------------------------- | ------- | ------- | ----------------------------------------------------- |
| `speculative.ngram_mod.n_match` | `int32` | `24`    | N-gram size for lookup (minimum 16 recommended)       |
| `speculative.ngram_mod.n_max`   | `int32` | `64`    | Maximum number of tokens to draft per step            |
| `speculative.ngram_mod.n_min`   | `int32` | `48`    | Minimum draft length required to attempt verification |

### Ngram-simple (`speculative.ngram_simple.*`)

Used with `speculative.type: ngram-simple`. A lightweight self-speculative method based on n-gram lookup with no draft model required.

| Param                               | Type    | Default | Description                                             |
| ----------------------------------- | ------- | ------- | ------------------------------------------------------- |
| `speculative.ngram_simple.size_n`   | `int32` | `12`    | N-gram size for lookup                                  |
| `speculative.ngram_simple.size_m`   | `int32` | `48`    | M-gram size for speculative tokens                      |
| `speculative.ngram_simple.min_hits` | `int32` | `1`     | Minimum hits at n-gram lookup for m-gram to be proposed |

### Ngram-map-k (`speculative.ngram_map_k.*`)

Used with `speculative.type: ngram-map-k`. Self-speculative decoding with n-gram keys only.

| Param                              | Type    | Default | Description                                             |
| ---------------------------------- | ------- | ------- | ------------------------------------------------------- |
| `speculative.ngram_map_k.size_n`   | `int32` | `12`    | N-gram size for lookup                                  |
| `speculative.ngram_map_k.size_m`   | `int32` | `48`    | M-gram size for speculative tokens                      |
| `speculative.ngram_map_k.min_hits` | `int32` | `1`     | Minimum hits at n-gram lookup for m-gram to be proposed |

### Ngram-map-k4v (`speculative.ngram_map_k4v.*`)

Used with `speculative.type: ngram-map-k4v`. Self-speculative decoding with n-gram keys and 4 m-gram values; higher acceptance rate than `ngram-map-k` at higher cost.

| Param                                | Type    | Default | Description                                             |
| ------------------------------------ | ------- | ------- | ------------------------------------------------------- |
| `speculative.ngram_map_k4v.size_n`   | `int32` | `12`    | N-gram size for lookup                                  |
| `speculative.ngram_map_k4v.size_m`   | `int32` | `48`    | M-gram size for speculative tokens                      |
| `speculative.ngram_map_k4v.min_hits` | `int32` | `1`     | Minimum hits at n-gram lookup for m-gram to be proposed |

### Ngram-cache (`speculative.ngram_cache.*`)

Used with `speculative.type: ngram-cache`. Self-speculative decoding with a 3-level n-gram cache; optionally load pre-built static and dynamic caches from disk.

| Param                                          | Type     | Default | Description                                    |
| ---------------------------------------------- | -------- | ------- | ---------------------------------------------- |
| `speculative.ngram_cache.lookup_cache_static`  | `string` | `""`    | Path to a pre-built static n-gram lookup cache |
| `speculative.ngram_cache.lookup_cache_dynamic` | `string` | `""`    | Path to a dynamic n-gram lookup cache file     |

## Prompt & Chat (`prompt.*`)

| Param                       | Type       | Default | Description                                                               |
| --------------------------- | ---------- | ------- | ------------------------------------------------------------------------- |
| `prompt.prefix`             | `string`   | `""`    | Text prepended to every user prompt                                       |
| `prompt.suffix`             | `string`   | `""`    | Text appended to every user prompt                                        |
| `prompt.system_prompt`      | `string`   | `""`    | Initial system prompt                                                     |
| `prompt.system_prompt_file` | `string`   | `""`    | Path to a file containing the system prompt                               |
| `prompt.system_prompt_type` | `string`   | `""`    | System prompt type (loads from a predefined YAML in `llama_ros/prompts/`) |
| `prompt.chat_template_file` | `string`   | `""`    | Path to a Jinja chat template file                                        |
| `prompt.stopping_words`     | `string[]` | `[]`    | List of words/tokens that stop generation                                 |

## LoRA Adapters (`lora.*`)

| Param                        | Type       | Default | Description                                         |
| ---------------------------- | ---------- | ------- | --------------------------------------------------- |
| `lora.adapters`              | `string[]` | `[]`    | List of LoRA adapter names to load                  |
| `lora.init_without_apply`    | `bool`     | `false` | Load LoRA adapters without applying them            |
| `lora.<lora_name>.repo`      | `string`   | `""`    | HuggingFace repository for the LoRA adapter         |
| `lora.<lora_name>.filename`  | `string`   | `""`    | Filename of the LoRA adapter in the repository      |
| `lora.<lora_name>.file_path` | `string`   | `""`    | Local file path to the LoRA adapter                 |
| `lora.<lora_name>.scale`     | `double`   | `1.0`   | LoRA adapter scale factor (clamped to `[0.0, 1.0]`) |
