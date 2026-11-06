# ROS 2 Interfaces

[← Back to README](../README.md)

## Messages and Actions

`llama_msgs` defines the messages and actions used by `llama_node` and `llava_node`. The tables below document the fields of each type. The service interfaces are documented in the [Services](#services) section.

### `SamplingConfig` (`llama_msgs/msg/SamplingConfig`)

The `SamplingConfig` message is used in `GenerateResponse` and `GenerateChatCompletions` goals to configure sampling behaviour per request.

| Field                      | Type               | Default                           | Description                                                                                                                                                                                                                    |
| -------------------------- | ------------------ | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `seed`                     | `uint32`           | `4294967295` (LLAMA_DEFAULT_SEED) | RNG seed (`4294967295` = random)                                                                                                                                                                                               |
| `n_prev`                   | `int32`            | `64`                              | Number of previous tokens to consider for repetition penalties                                                                                                                                                                 |
| `n_probs`                  | `int32`            | `0`                               | Return top-N token probabilities (`0` = disabled)                                                                                                                                                                              |
| `min_keep`                 | `int32`            | `0`                               | Minimum number of tokens to keep after sampling (`0` = disabled)                                                                                                                                                               |
| `ignore_eos`               | `bool`             | `false`                           | Ignore end-of-stream tokens and continue generating                                                                                                                                                                            |
| `no_perf`                  | `bool`             | `false`                           | Disable performance metrics collection                                                                                                                                                                                         |
| `timing_per_token`         | `bool`             | `false`                           | Collect per-token timing data                                                                                                                                                                                                  |
| `logit_bias`               | `LogitBiasArray`   | `[]`                              | Logit biases for specific tokens                                                                                                                                                                                               |
| `logit_bias_eog`           | `LogitBiasArray`   | `[]`                              | Pre-calculated logit biases for end-of-generation tokens                                                                                                                                                                       |
| `temp`                     | `float32`          | `0.80`                            | Sampling temperature                                                                                                                                                                                                           |
| `dynatemp_range`           | `float32`          | `0.0`                             | Dynamic temperature range (`0.0` = disabled)                                                                                                                                                                                   |
| `dynatemp_exponent`        | `float32`          | `1.0`                             | Dynamic temperature exponent                                                                                                                                                                                                   |
| `top_k`                    | `int32`            | `40`                              | Top-K sampling (`0` = disabled)                                                                                                                                                                                                |
| `top_p`                    | `float32`          | `0.95`                            | Top-P (nucleus) sampling (`1.0` = disabled)                                                                                                                                                                                    |
| `min_p`                    | `float32`          | `0.05`                            | Min-P sampling (`0.0` = disabled)                                                                                                                                                                                              |
| `top_n_sigma`              | `float32`          | `-1.0`                            | Top-N-sigma sampling (`-1.0` = disabled)                                                                                                                                                                                       |
| `xtc_probability`          | `float32`          | `0.0`                             | XTC sampling probability (`0.0` = disabled)                                                                                                                                                                                    |
| `xtc_threshold`            | `float32`          | `0.1`                             | XTC sampling threshold (values > `0.5` disable XTC)                                                                                                                                                                            |
| `typical_p`                | `float32`          | `1.0`                             | Locally typical sampling (`1.0` = disabled)                                                                                                                                                                                    |
| `penalty_last_n`           | `int32`            | `64`                              | Number of last tokens to consider for penalties (`0` = disable, `-1` = context size)                                                                                                                                           |
| `penalty_repeat`           | `float32`          | `1.0`                             | Repetition penalty (`1.0` = disabled)                                                                                                                                                                                          |
| `penalty_freq`             | `float32`          | `0.0`                             | Frequency penalty (`0.0` = disabled)                                                                                                                                                                                           |
| `penalty_present`          | `float32`          | `0.0`                             | Presence penalty (`0.0` = disabled)                                                                                                                                                                                            |
| `dry_multiplier`           | `float32`          | `0.0`                             | DRY repetition penalty multiplier (`0.0` = disabled)                                                                                                                                                                           |
| `dry_base`                 | `float32`          | `1.75`                            | DRY repetition penalty base                                                                                                                                                                                                    |
| `dry_allowed_length`       | `int32`            | `2`                               | Tokens extending repetitions beyond this length receive DRY penalty                                                                                                                                                            |
| `dry_penalty_last_n`       | `int32`            | `64`                              | Tokens to scan for DRY repetitions (`0` = disable, `-1` = context size)                                                                                                                                                        |
| `dry_sequence_breakers`    | `string[]`         | `["\n", ":", "\"", "*"]`          | Sequence breakers for DRY                                                                                                                                                                                                      |
| `adaptive_target`          | `float32`          | `-1.0`                            | Adaptive-P target probability (negative = disabled)                                                                                                                                                                            |
| `adaptive_decay`           | `float32`          | `0.90`                            | Adaptive-P EMA decay                                                                                                                                                                                                           |
| `mirostat`                 | `int32`            | `0`                               | Mirostat mode (`0` = disabled, `1` = Mirostat v1, `2` = Mirostat v2)                                                                                                                                                           |
| `mirostat_eta`             | `float32`          | `0.10`                            | Mirostat learning rate                                                                                                                                                                                                         |
| `mirostat_tau`             | `float32`          | `5.0`                             | Mirostat target entropy                                                                                                                                                                                                        |
| `samplers_sequence`        | `string`           | `"edskypmxt"`                     | Sampler pipeline order (chars map to: `e`=penalties, `d`=DRY, `s`=top-N-sigma, `k`=top-K, `y`=typical-P, `p`=top-P, `m`=min-P, `x`=XTC, `t`=temp)                                                                              |
| `grammar`                  | `string`           | `""`                              | GBNF grammar string to constrain sampling                                                                                                                                                                                      |
| `grammar_schema`           | `string`           | `""`                              | JSON schema converted to a GBNF grammar                                                                                                                                                                                        |
| `grammar_lazy`             | `bool`             | `false`                           | Use lazy grammar evaluation (grammar only activates after a trigger)                                                                                                                                                           |
| `grammar_triggers`         | `GrammarTrigger[]` | `[]`                              | Trigger conditions for lazy grammar activation                                                                                                                                                                                 |
| `preserved_tokens`         | `int32[]`          | `[]`                              | Token IDs that should never be penalised or modified                                                                                                                                                                           |
| `backend_sampling`         | `bool`             | `false`                           | Use hardware-accelerated (backend) sampling if available                                                                                                                                                                       |
| `reasoning_budget`         | `int32`            | `-1`                              | Token budget for reasoning (`-1` = disabled, `≥ 0` = max thinking tokens)                                                                                                                                                      |
| `reasoning_budget_start`   | `int32[]`          | `[]`                              | Token IDs for the thinking start tag. Auto-populated from the chat template if empty                                                                                                                                           |
| `reasoning_budget_end`     | `int32[]`          | `[]`                              | Token IDs for the thinking end tag. Auto-populated from the chat template if empty                                                                                                                                             |
| `reasoning_budget_forced`  | `int32[]`          | `[]`                              | Token sequence forcibly injected when the budget is exhausted (message + end tag). Auto-populated if empty. Distinct from `force_pure_content_parser`: this controls _when_ to stop thinking, not _how_ the template is parsed |
| `reasoning_budget_message` | `string`           | `""`                              | Text inserted before the thinking end tag when the reasoning budget is exhausted (e.g. `"Wait, I need to conclude."`)                                                                                                          |
| `reasoning_control`        | `bool`             | `false`                           | Create the budget sampler on demand so reasoning can be ended at runtime                                                                                                                                                       |

### `LogitBias` (`llama_msgs/msg/LogitBias`)

| Field   | Type      | Default | Description                     |
| ------- | --------- | ------- | ------------------------------- |
| `token` | `int32`   | —       | Token to apply the bias to      |
| `bias`  | `float32` | —       | Bias added to the token's logit |

### `LogitBiasArray` (`llama_msgs/msg/LogitBiasArray`)

| Field  | Type          | Default | Description          |
| ------ | ------------- | ------- | -------------------- |
| `data` | `LogitBias[]` | `[]`    | List of logit biases |

### `GrammarTrigger` (`llama_msgs/msg/GrammarTrigger`)

| Constant                             | Value |
| ------------------------------------ | ----- |
| `GRAMMAR_TRIGGER_TYPE_TOKEN`         | `0`   |
| `GRAMMAR_TRIGGER_TYPE_WORD`          | `1`   |
| `GRAMMAR_TRIGGER_TYPE_PATTERN`       | `2`   |
| `GRAMMAR_TRIGGER_TYPE_PATTERN_START` | `3`   |

| Field   | Type     | Default | Description                               |
| ------- | -------- | ------- | ----------------------------------------- |
| `type`  | `int32`  | `0`     | Trigger type (one of the constants above) |
| `value` | `string` | `""`    | Trigger word or pattern                   |
| `token` | `int32`  | `0`     | Trigger token (for token triggers)        |

### `ChatMessage` (`llama_msgs/msg/ChatMessage`)

| Field               | Type             | Default | Description                                                 |
| ------------------- | ---------------- | ------- | ----------------------------------------------------------- |
| `role`              | `string`         | `""`    | Message role (`system`, `user`, `assistant`, `tool`)        |
| `content`           | `string`         | `""`    | Message content                                             |
| `content_parts`     | `ChatContent[]`  | `[]`    | Content split into parts (used for multimodal placeholders) |
| `tool_calls`        | `ChatToolCall[]` | `[]`    | Tool calls made by the assistant                            |
| `reasoning_content` | `string`         | `""`    | Reasoning content extracted from the model output           |
| `tool_name`         | `string`         | `""`    | Name of the required tool                                   |
| `tool_call_id`      | `string`         | `""`    | ID of the tool call                                         |

### `ChatContent` (`llama_msgs/msg/ChatContent`)

| Field  | Type     | Default | Description                       |
| ------ | -------- | ------- | --------------------------------- |
| `type` | `string` | `""`    | Part type (`text`, image marker…) |
| `text` | `string` | `""`    | Part text                         |

### `ChatToolCall` (`llama_msgs/msg/ChatToolCall`)

| Field       | Type     | Default | Description                  |
| ----------- | -------- | ------- | ---------------------------- |
| `name`      | `string` | `""`    | Tool name                    |
| `arguments` | `string` | `""`    | Tool arguments (JSON string) |
| `id`        | `string` | `""`    | Tool call ID                 |
| `index`     | `int32`  | `0`     | Tool call index              |

### `ChatReqTool` (`llama_msgs/msg/ChatReqTool`)

| Field      | Type       | Default      | Description                 |
| ---------- | ---------- | ------------ | --------------------------- |
| `type`     | `string`   | `"function"` | Tool type (only `function`) |
| `function` | `ChatTool` | —            | Tool definition             |

### `ChatTool` (`llama_msgs/msg/ChatTool`)

| Constant               | Value |
| ---------------------- | ----- |
| `TOOL_CHOICE_AUTO`     | `0`   |
| `TOOL_CHOICE_REQUIRED` | `1`   |
| `TOOL_CHOICE_NONE`     | `2`   |

| Field         | Type     | Default | Description                   |
| ------------- | -------- | ------- | ----------------------------- |
| `name`        | `string` | `""`    | Tool name                     |
| `description` | `string` | `""`    | Tool description              |
| `parameters`  | `string` | `""`    | Tool parameters (JSON schema) |

### `ChatReasoningFormat` (`llama_msgs/msg/ChatReasoningFormat`)

| Constant                                  | Value |
| ----------------------------------------- | ----- |
| `COMMON_REASONING_FORMAT_NONE`            | `0`   |
| `COMMON_REASONING_FORMAT_AUTO`            | `1`   |
| `COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY` | `2`   |
| `COMMON_REASONING_FORMAT_DEEPSEEK`        | `3`   |

| Field   | Type    | Default | Description                                   |
| ------- | ------- | ------- | --------------------------------------------- |
| `value` | `int32` | `0`     | Reasoning format (one of the constants above) |

### `Metadata` (`llama_msgs/msg/Metadata`)

Returned by the `get_metadata` service (available for every model type).

| Field       | Type            | Default | Description                           |
| ----------- | --------------- | ------- | ------------------------------------- |
| `general`   | `GeneralInfo`   | —       | General model info                    |
| `model`     | `ModelInfo`     | —       | Architecture info                     |
| `tokenizer` | `TokenizerInfo` | —       | Tokenizer info                        |
| `sampling`  | `SamplingInfo`  | —       | Sampling defaults stored in the model |
| `decision`  | `DecisionInfo`  | —       | Decision-model info                   |

### `GeneralInfo` (`llama_msgs/msg/GeneralInfo`)

| Field                  | Type              | Default | Description                                      |
| ---------------------- | ----------------- | ------- | ------------------------------------------------ |
| `architecture`         | `string`          | `""`    | Architecture the model implements                |
| `quantization_version` | `uint32`          | `0`     | Quantization format version (0 if not quantized) |
| `alignment`            | `uint32`          | `0`     | Global alignment                                 |
| `name`                 | `string`          | `""`    | Human-readable model name                        |
| `author`               | `string`          | `""`    | Model author                                     |
| `version`              | `string`          | `""`    | Model version                                    |
| `organization`         | `string`          | `""`    | Model organization                               |
| `basename`             | `string`          | `""`    | Base model name / architecture                   |
| `finetune`             | `string`          | `""`    | What the base model was optimized toward         |
| `description`          | `string`          | `""`    | Free-form description                            |
| `quantized_by`         | `string`          | `""`    | Who quantized the model                          |
| `size_label`           | `string`          | `""`    | Size class (e.g. `4B`, `270M`)                   |
| `license`              | `string`          | `""`    | License identifier                               |
| `license_name`         | `string`          | `""`    | Human-friendly license name                      |
| `license_link`         | `string`          | `""`    | URL to the license                               |
| `url`                  | `string`          | `""`    | Model homepage URL                               |
| `repo_url`             | `string`          | `""`    | Model repository URL                             |
| `doi`                  | `string`          | `""`    | Digital Object Identifier                        |
| `uuid`                 | `string`          | `""`    | Universally unique identifier                    |
| `file_type`            | `string`          | `""`    | Majority tensor type of the file                 |
| `tags`                 | `string[]`        | `[]`    | Tags associated with the model                   |
| `languages`            | `string[]`        | `[]`    | Languages supported by the model                 |
| `datasets`             | `string[]`        | `[]`    | Datasets used to train the model                 |
| `base_models`          | `BaseModelInfo[]` | `[]`    | Base models this model derives from              |

### `ModelInfo` (`llama_msgs/msg/ModelInfo`)

| Field                   | Type            | Default | Description                                    |
| ----------------------- | --------------- | ------- | ---------------------------------------------- |
| `context_length`        | `uint64`        | `0`     | Training context length in tokens              |
| `embedding_length`      | `uint64`        | `0`     | Embedding layer size                           |
| `block_count`           | `uint64`        | `0`     | Number of attention + feed-forward blocks      |
| `feed_forward_length`   | `uint64`        | `0`     | Feed-forward layer length                      |
| `use_parallel_residual` | `bool`          | `false` | Whether parallel residual connections are used |
| `tensor_data_layout`    | `string`        | `""`    | Tensor data layout chosen during conversion    |
| `expert_count`          | `uint32`        | `0`     | Number of experts (MoE models)                 |
| `expert_used_count`     | `uint32`        | `0`     | Experts used per token (MoE models)            |
| `attention`             | `AttentionInfo` | —       | Attention info                                 |
| `rope`                  | `RoPEInfo`      | —       | RoPE info                                      |

### `AttentionInfo` (`llama_msgs/msg/AttentionInfo`)

| Field                    | Type      | Default | Description                     |
| ------------------------ | --------- | ------- | ------------------------------- |
| `head_count`             | `uint64`  | `0`     | Number of attention heads       |
| `head_count_kv`          | `uint64`  | `0`     | Number of key/value heads (GQA) |
| `max_alibi_bias`         | `float32` | `0.0`   | Maximum ALiBI bias              |
| `clamp_kqv`              | `float32` | `0.0`   | Clamp value for Q, K and V      |
| `layer_norm_epsilon`     | `float32` | `0.0`   | Layer normalization epsilon     |
| `layer_norm_rms_epsilon` | `float32` | `0.0`   | RMS normalization epsilon       |
| `key_length`             | `uint32`  | `0`     | Key head size (optional)        |
| `value_length`           | `uint32`  | `0`     | Value head size (optional)      |

### `RoPEInfo` (`llama_msgs/msg/RoPEInfo`)

| Field                             | Type      | Default | Description                                   |
| --------------------------------- | --------- | ------- | --------------------------------------------- |
| `dimension_count`                 | `uint64`  | `0`     | Number of rotary dimensions                   |
| `freq_base`                       | `float32` | `0.0`   | RoPE base frequency                           |
| `scaling_type`                    | `string`  | `""`    | Scaling type (`none`, `linear`, `yarn`)       |
| `scaling_factor`                  | `float32` | `0.0`   | RoPE context scaling factor                   |
| `scaling_original_context_length` | `uint32`  | `0`     | Original context length of the base model     |
| `scaling_finetuned`               | `bool`    | `false` | Whether the model was fine-tuned with scaling |

### `TokenizerInfo` (`llama_msgs/msg/TokenizerInfo`)

| Field                | Type       | Default | Description                                |
| -------------------- | ---------- | ------- | ------------------------------------------ |
| `model`              | `string`   | `""`    | Tokenizer model name                       |
| `bos_token_id`       | `uint32`   | `0`     | Beginning-of-sequence token id             |
| `eos_token_id`       | `uint32`   | `0`     | End-of-sequence token id                   |
| `unknown_token_id`   | `uint32`   | `0`     | Unknown token id                           |
| `padding_token_id`   | `uint32`   | `0`     | Padding token id                           |
| `separator_token_id` | `uint32`   | `0`     | Separator token id                         |
| `add_bos_token`      | `bool`     | `false` | Whether a BOS token is added               |
| `chat_template`      | `string`   | `""`    | Default chat template (untruncated)        |
| `add_eos_token`      | `bool`     | `false` | Whether an EOS token is added              |
| `mask_token_id`      | `uint32`   | `0`     | Mask token id (0 if absent)                |
| `chat_templates`     | `string[]` | `[]`    | Named chat templates, e.g. `["systemone"]` |

### `BaseModelInfo` (`llama_msgs/msg/BaseModelInfo`)

| Field          | Type     | Default | Description               |
| -------------- | -------- | ------- | ------------------------- |
| `name`         | `string` | `""`    | Base model name           |
| `author`       | `string` | `""`    | Base model author         |
| `version`      | `string` | `""`    | Base model version        |
| `organization` | `string` | `""`    | Base model organization   |
| `repo_url`     | `string` | `""`    | Base model repository URL |

### `SamplingInfo` (`llama_msgs/msg/SamplingInfo`)

Sampling defaults stored in the GGUF metadata (`general.sampling.*`). Absent
keys are reported as `0`/empty.

| Field             | Type       | Default | Description                                  |
| ----------------- | ---------- | ------- | -------------------------------------------- |
| `sequence`        | `string[]` | `[]`    | Recommended sampler sequence                 |
| `top_k`           | `int32`    | `0`     | Top-K sampling (0 = disabled)                |
| `top_p`           | `float32`  | `0.0`   | Top-P sampling (0.0 = not declared)          |
| `min_p`           | `float32`  | `0.0`   | Min-P sampling (0.0 = disabled)              |
| `xtc_probability` | `float32`  | `0.0`   | XTC probability (0.0 = disabled)             |
| `xtc_threshold`   | `float32`  | `0.0`   | XTC threshold                                |
| `temp`            | `float32`  | `0.0`   | Temperature                                  |
| `penalty_last_n`  | `int32`    | `0`     | Repetition penalty window                    |
| `penalty_repeat`  | `float32`  | `0.0`   | Repetition penalty                           |
| `mirostat`        | `int32`    | `0`     | Mirostat mode (0 = disabled, 1 = v1, 2 = v2) |
| `mirostat_tau`    | `float32`  | `0.0`   | Mirostat tau                                 |
| `mirostat_eta`    | `float32`  | `0.0`   | Mirostat eta                                 |

### `DecisionInfo` (`llama_msgs/msg/DecisionInfo`)

Tells whether the node exposes the `evaluate_decisions` service and which
upstream decision type the model declares.

| Field                | Type        | Default | Description                                                         |
| -------------------- | ----------- | ------- | ------------------------------------------------------------------- |
| `enabled`            | `bool`      | `false` | Whether `evaluate_decisions` is exposed                             |
| `type`               | `string`    | `""`    | `openjev`, `lev`, `kev`, `nimble`, `laya` or `clef` (empty if none) |
| `max_head_tokens`    | `uint32`    | `0`     | Decision max head tokens (0 if absent)                              |
| `temperature_names`  | `string[]`  | `[]`    | Temperature names, e.g. `"choice"`, `"choice.3_5"`                  |
| `temperatures`       | `float32[]` | `[]`    | Temperature values, aligned with `temperature_names`                |
| `systemone_template` | `string`    | `""`    | `systemone` chat template (empty if absent)                         |

### `DecisionQuestion` (`llama_msgs/msg/DecisionQuestion`)

One typed question of an `evaluate_decisions` request.

| Constant | Value |
| -------- | ----- |
| `CHOICE` | `0`   |
| `SCORE`  | `1`   |
| `NOUL`   | `2`   |

| Field          | Type       | Default | Description                                                            |
| -------------- | ---------- | ------- | ---------------------------------------------------------------------- |
| `type`         | `uint8`    | `0`     | Question type (one of the constants above)                             |
| `instructions` | `string`   | `""`    | Question instructions                                                  |
| `keys`         | `string[]` | `[]`    | Choice option keys (ignored otherwise)                                 |
| `descriptions` | `string[]` | `[]`    | Choice option descriptions; score levels (2..10); Noul `[false, true]` |

### `DecisionAnswer` (`llama_msgs/msg/DecisionAnswer`)

One answer of an `evaluate_decisions` response, aligned 1:1 with the request
questions.

| Field           | Type        | Default | Description                                       |
| --------------- | ----------- | ------- | ------------------------------------------------- |
| `success`       | `bool`      | `false` | Whether the question was answered                 |
| `error`         | `string`    | `""`    | Error message when `success` is false             |
| `type`          | `uint8`     | `0`     | Question type (see `DecisionQuestion`)            |
| `choice`        | `string`    | `""`    | Best option key (choice questions)                |
| `score`         | `float32`   | `0.0`   | Expected level (score questions)                  |
| `noul`          | `float32`   | `0.0`   | P(true) (Noul questions)                          |
| `confidence`    | `float32`   | `0.0`   | Answer confidence (0 for Noul)                    |
| `keys`          | `string[]`  | `[]`    | Ordered option keys, aligned with `probabilities` |
| `probabilities` | `float32[]` | `[]`    | Probability of each option                        |

### `Response` (`llama_msgs/msg/Response`)

| Field    | Type               | Default | Description                       |
| -------- | ------------------ | ------- | --------------------------------- |
| `text`   | `string`           | `""`    | Generated text                    |
| `tokens` | `int32[]`          | `[]`    | Generated token ids               |
| `probs`  | `TokenProbArray[]` | `[]`    | Probabilities per generated token |

### `PartialResponse` (`llama_msgs/msg/PartialResponse`)

| Field   | Type             | Default | Description                                            |
| ------- | ---------------- | ------- | ------------------------------------------------------ |
| `text`  | `string`         | `""`    | Delta text of the streamed response                    |
| `token` | `int32`          | `0`     | Token of the delta                                     |
| `probs` | `TokenProbArray` | —       | Probabilities of the selected and more probable tokens |

### `TokenProb` (`llama_msgs/msg/TokenProb`)

| Field         | Type      | Default | Description                      |
| ------------- | --------- | ------- | -------------------------------- |
| `token`       | `int32`   | `0`     | Token id                         |
| `probability` | `float32` | `0.0`   | Probability of the token         |
| `token_text`  | `string`  | `""`    | Text representation of the token |

### `TokenProbArray` (`llama_msgs/msg/TokenProbArray`)

| Field          | Type          | Default | Description                               |
| -------------- | ------------- | ------- | ----------------------------------------- |
| `data`         | `TokenProb[]` | `[]`    | Probabilities of the most probable tokens |
| `chosen_token` | `int32`       | `0`     | Chosen token in the sampling              |

### `UsageStats` (`llama_msgs/msg/UsageStats`)

| Field               | Type    | Default | Description                |
| ------------------- | ------- | ------- | -------------------------- |
| `completion_tokens` | `int32` | `0`     | Number of output tokens    |
| `prompt_tokens`     | `int32` | `0`     | Number of input tokens     |
| `total_tokens`      | `int32` | `0`     | Prompt + completion tokens |

### `GenerateChatCompletions` Goal (`llama_msgs/action/GenerateChatCompletions`)

| Field                       | Type                         | Default | Description                                                                                     |
| --------------------------- | ---------------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `messages`                  | `ChatMessage[]`              | `[]`    | Conversation history as a list of chat messages                                                 |
| `add_generation_prompt`     | `bool`                       | `false` | Append the generation prompt token sequence after the last message                              |
| `use_jinja`                 | `bool`                       | `false` | Use Jinja2 chat template (required for tool calls and reasoning)                                |
| `tools`                     | `ChatReqTool[]`              | `[]`    | List of tools the model may call                                                                |
| `tool_choice`               | `int32`                      | `0`     | Tool selection mode: `0` = auto, `1` = required (must call a tool), `2` = none                  |
| `extract_reasoning`         | `bool`                       | `false` | Extract `<think>` reasoning content from the response into `reasoning_content`                  |
| `sampling_config`           | `SamplingConfig`             | —       | Per-request sampling configuration (see `SamplingConfig` table above)                           |
| `reasoning_format`          | `ChatReasoningFormat`        | `0`     | How reasoning content is returned: `0`=none, `1`=auto, `2`=deepseek_legacy, `3`=deepseek        |
| `images`                    | `sensor_msgs/Image[]`        | `[]`    | Images for VLM inference                                                                        |
| `audios`                    | `std_msgs/UInt8MultiArray[]` | `[]`    | Audio buffers for multimodal inference                                                          |
| `parallel_tool_calls`       | `bool`                       | `false` | Allow the model to return multiple tool calls in a single message                               |
| `stream`                    | `bool`                       | `false` | Stream partial results as feedback messages                                                     |
| `force_pure_content_parser` | `bool`                       | `false` | Bypass Jinja template tool-call/reasoning parsing and force raw content output for this request |

### `GenerateChatCompletions` Result (`llama_msgs/action/GenerateChatCompletions`)

| Field                | Type           | Default             | Description               |
| -------------------- | -------------- | ------------------- | ------------------------- |
| `id`                 | `string`       | `""`                | Unique chat completion ID |
| `choices`            | `ChatChoice[]` | `[]`                | List of chat completions  |
| `created`            | `int32`        | `0`                 | Creation time             |
| `model`              | `string`       | `""`                | Model used                |
| `system_fingerprint` | `string`       | `""`                | System fingerprint        |
| `object`             | `string`       | `"chat.completion"` | Object type               |
| `usage`              | `UsageStats`   | —                   | Usage statistics          |

### `ChatChoice` (`llama_msgs/msg/ChatChoice`)

| Field           | Type               | Default | Description                 |
| --------------- | ------------------ | ------- | --------------------------- |
| `message`       | `ChatMessage`      | —       | The message that was sent   |
| `logprobs`      | `TokenProbArray[]` | `[]`    | Log probabilities           |
| `finish_reason` | `string`           | `""`    | Reason the completion ended |
| `index`         | `int32`            | `0`     | Choice index                |

### `GenerateChatCompletions` Feedback (`llama_msgs/action/GenerateChatCompletions`)

| Field                | Type                | Default                   | Description               |
| -------------------- | ------------------- | ------------------------- | ------------------------- |
| `id`                 | `string`            | `""`                      | Unique chat completion ID |
| `choices`            | `ChatChoiceChunk[]` | `[]`                      | List of chat chunks       |
| `created`            | `int32`             | `0`                       | Creation time             |
| `model`              | `string`            | `""`                      | Model used                |
| `system_fingerprint` | `string`            | `""`                      | System fingerprint        |
| `object`             | `string`            | `"chat.completion.chunk"` | Object type               |
| `usage`              | `UsageStats`        | —                         | Usage statistics          |

### `ChatChoiceChunk` (`llama_msgs/msg/ChatChoiceChunk`)

| Field           | Type             | Default | Description                 |
| --------------- | ---------------- | ------- | --------------------------- |
| `delta`         | `ChatDeltaChunk` | —       | Delta applied to chat state |
| `logprobs`      | `TokenProbArray` | —       | Log probabilities           |
| `finish_reason` | `string`         | `""`    | Reason the completion ended |
| `index`         | `int32`          | `0`     | Choice index                |

### `ChatDeltaChunk` (`llama_msgs/msg/ChatDeltaChunk`)

| Field               | Type             | Default | Description                    |
| ------------------- | ---------------- | ------- | ------------------------------ |
| `content`           | `string`         | `""`    | Delta content                  |
| `role`              | `string`         | `""`    | Delta role                     |
| `tool_calls`        | `ChatToolCall[]` | `[]`    | Tool calls in the delta        |
| `reasoning_content` | `string`         | `""`    | Reasoning content in the delta |

### `GenerateResponse` Goal (`llama_msgs/action/GenerateResponse`)

| Field             | Type                         | Default | Description                                               |
| ----------------- | ---------------------------- | ------- | --------------------------------------------------------- |
| `prompt`          | `string`                     | `""`    | Prompt                                                    |
| `images`          | `sensor_msgs/Image[]`        | `[]`    | Images for VLMs                                           |
| `audios`          | `std_msgs/UInt8MultiArray[]` | `[]`    | Audios for mtmd                                           |
| `stop`            | `string[]`                   | `[]`    | Stop sequences                                            |
| `precompute`      | `bool`                       | `false` | Evaluate the prompt and retain KV without sampling output |
| `reset`           | `bool`                       | `false` | Whether to reset the context                              |
| `sampling_config` | `SamplingConfig`             | —       | Per-request sampling configuration                        |

### `GenerateResponse` Result (`llama_msgs/action/GenerateResponse`)

| Field      | Type       | Default | Description    |
| ---------- | ---------- | ------- | -------------- |
| `response` | `Response` | —       | Final response |

### `GenerateResponse` Feedback (`llama_msgs/action/GenerateResponse`)

| Field              | Type              | Default | Description      |
| ------------------ | ----------------- | ------- | ---------------- |
| `partial_response` | `PartialResponse` | —       | Partial response |

### `LoRA` (`llama_msgs/msg/LoRA`)

| Field   | Type      | Default | Description                  |
| ------- | --------- | ------- | ---------------------------- |
| `id`    | `int32`   | `0`     | LoRA ID                      |
| `path`  | `string`  | `""`    | Path to the LoRA file        |
| `scale` | `float32` | `0.0`   | Scale applied to the adapter |

## Services

`llama_msgs` also defines the services exposed by `llama_node` and `llava_node` under the node namespace (e.g. `/llama/tokenize`).

### `Tokenize` (`llama_msgs/srv/Tokenize`)

**Request**

| Field  | Type     | Default | Description      |
| ------ | -------- | ------- | ---------------- |
| `text` | `string` | `""`    | Text to tokenize |

**Response**

| Field    | Type      | Default | Description           |
| -------- | --------- | ------- | --------------------- |
| `tokens` | `int32[]` | `[]`    | Token ids of the text |

### `Detokenize` (`llama_msgs/srv/Detokenize`)

**Request**

| Field    | Type      | Default | Description             |
| -------- | --------- | ------- | ----------------------- |
| `tokens` | `int32[]` | `[]`    | Token ids to detokenize |

**Response**

| Field  | Type     | Default | Description                  |
| ------ | -------- | ------- | ---------------------------- |
| `text` | `string` | `""`    | Text decoded from the tokens |

### `GenerateEmbeddings` (`llama_msgs/srv/GenerateEmbeddings`)

**Request**

| Field           | Type                         | Default | Description                                                                               |
| --------------- | ---------------------------- | ------- | ----------------------------------------------------------------------------------------- |
| `prompt`        | `string`                     | `""`    | Prompt; use `<__media__>` as a placeholder for each image/audio                           |
| `images`        | `sensor_msgs/Image[]`        | `[]`    | Images for multimodal embedding models                                                    |
| `audios`        | `std_msgs/UInt8MultiArray[]` | `[]`    | Audios for multimodal embedding models                                                    |
| `normalization` | `int32`                      | `2`     | Normalization: `-1`=none, `0`=max absolute int16, `1`=taxicab, `2`=euclidean, `>2`=p-norm |

**Response**

| Field        | Type        | Default | Description                |
| ------------ | ----------- | ------- | -------------------------- |
| `embeddings` | `float32[]` | `[]`    | Generated embeddings       |
| `n_tokens`   | `int32`     | `0`     | Number of tokens processed |

### `RerankDocuments` (`llama_msgs/srv/RerankDocuments`)

**Request**

| Field       | Type       | Default | Description                          |
| ----------- | ---------- | ------- | ------------------------------------ |
| `query`     | `string`   | `""`    | Query to score the documents against |
| `documents` | `string[]` | `[]`    | Documents to rerank                  |

**Response**

| Field    | Type        | Default | Description                        |
| -------- | ----------- | ------- | ---------------------------------- |
| `scores` | `float32[]` | `[]`    | Relevance scores for the documents |

### `GetMetadata` (`llama_msgs/srv/GetMetadata`)

This service has an empty request.

**Response**

| Field      | Type       | Default | Description                          |
| ---------- | ---------- | ------- | ------------------------------------ |
| `metadata` | `Metadata` | —       | Metadata info (see `Metadata` above) |

### `ListLoRAs` (`llama_msgs/srv/ListLoRAs`)

This service has an empty request.

**Response**

| Field   | Type     | Default | Description                           |
| ------- | -------- | ------- | ------------------------------------- |
| `loras` | `LoRA[]` | `[]`    | LoRAs loaded when launching llama_ros |

### `UpdateLoRAs` (`llama_msgs/srv/UpdateLoRAs`)

**Request**

| Field   | Type     | Default | Description                                             |
| ------- | -------- | ------- | ------------------------------------------------------- |
| `loras` | `LoRA[]` | `[]`    | LoRAs to update. A LoRA with scale `0.0` deactivates it |

This service has an empty response.

### `EvaluateDecisions` (`llama_msgs/srv/EvaluateDecisions`)

**Request**

| Field       | Type                  | Default | Description                                      |
| ----------- | --------------------- | ------- | ------------------------------------------------ |
| `state`     | `string`              | `""`    | One state shared by all questions (JSON or text) |
| `images`    | `sensor_msgs/Image[]` | `[]`    | Optional images (OpenJEV and Clef)               |
| `questions` | `DecisionQuestion[]`  | `[]`    | Typed questions (must be non-empty)              |

**Response**

| Field     | Type               | Default | Description                            |
| --------- | ------------------ | ------- | -------------------------------------- |
| `answers` | `DecisionAnswer[]` | `[]`    | Answers aligned 1:1 with the questions |
