uint64 context_length                # Length of the context (in tokens) that the model was trained on. For most architectures, this is the hard limit on the length of the input
uint64 embedding_length              # Embedding layer size
uint64 block_count                   # The number of blocks of attention+feed-forward layers (i.e. the bulk of the LLM). Does not include the input or embedding layers
uint64 feed_forward_length           # The length of the feed-forward layer

bool use_parallel_residual           # Whether or not the parallel residual logic should be used
string tensor_data_layout            # When a model is converted to GGUF, tensors may be rearranged to improve performance. This key describes the layout of the tensor data

uint32 expert_count                  # Number of experts in MoE models (optional for non-MoE arches)
uint32 expert_used_count             # Number of experts used during each token token evaluation (optional for non-MoE arches)

AttentionInfo attention              # Attention info for the model
RoPEInfo rope                        # RoPE info for the model