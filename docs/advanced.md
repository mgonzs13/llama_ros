# Advanced Topics

[← Back to README](../README.md)

## LoRA Adapters

You can use LoRA adapters when launching LLMs. Using llama.cpp features, you can load multiple adapters choosing the scale to apply for each adapter. Here you have an example of using LoRA adapters with Phi-3. You can list the
LoRAs using the `/llama/list_loras` service and modify their scales values by using the `/llama/update_loras` service. A scale value of 0.0 means not using that LoRA.

<details>
<summary>Click to expand</summary>

```yaml
/**:
  ros__parameters:
    model:
      repo: bartowski/Phi-3.5-mini-instruct-GGUF
      filename: Phi-3.5-mini-instruct-Q4_K_M.gguf
    context:
      n_ctx: 2048
      n_batch: 8
      n_ubatch: 8
      n_predict: 2048
    gpu:
      n_gpu_layers: 0
    cpu:
      n_threads: 1
    prompt:
      system_prompt_type: Phi-3
    lora:
      adapters:
        - code_writer
        - summarization
      code_writer:
        repo: zhhan/adapter-Phi-3-mini-4k-instruct_code_writing
        filename: Phi-3-mini-4k-instruct-adaptor-f16-code_writer.gguf
        scale: 0.5
      summarization:
        repo: zhhan/adapter-Phi-3-mini-4k-instruct_summarization
        filename: Phi-3-mini-4k-instruct-adaptor-f16-summarization.gguf
        scale: 0.5
```

</details>

## See Also

- [Speculative decoding parameters](parameters.md#speculative-decoding-speculative)
- [Prompt and chat parameters](parameters.md#prompt--chat-prompt)
- [Decision demos](demos.md#decision-demo)
- [Embeddings generation demo](demos.md#embeddings-generation-demo)
- [Reranking demo](demos.md#reranking-demo)
