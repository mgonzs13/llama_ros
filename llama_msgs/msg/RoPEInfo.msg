uint64 dimension_count                      # The number of rotary dimensions for RoPE
float32 freq_base                           # The base frequency for RoPE

string scaling_type                         # Can be none, linear, or yarn.
float32 scaling_factor                      # A scale factor for RoPE to adjust the context length.
uint32 scaling_original_context_length      # The original context length of the base model.
bool scaling_finetuned                      # True if model has been finetuned with RoPE scaling.