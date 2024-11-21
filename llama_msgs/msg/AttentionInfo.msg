uint64 head_count                   # Number of attention heads (n_head)
uint64 head_count_kv                # Number of heads per group for GQA

float32 max_alibi_bias              # Maximum bias for ALiBI
float32 clamp_kqv                   # Clamp value for Q, K, and V tensors

float32 layer_norm_epsilon          # Epsilon for layer normalization
float32 layer_norm_rms_epsilon      # Epsilon for RMS normalization

uint32 key_length                   # Size of a key head (optional)
uint32 value_length                 # Size of a value head (optional)