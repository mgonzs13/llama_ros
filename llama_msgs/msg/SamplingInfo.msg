string[] sequence         # Recommended sampler sequence
int32 top_k               # Top-k sampling (0 = disabled)
float32 top_p             # Top-p sampling (0.0 = not declared)
float32 min_p             # Min-p sampling (0.0 = disabled)
float32 xtc_probability   # XTC probability (0.0 = disabled)
float32 xtc_threshold     # XTC threshold
float32 temp              # Temperature
int32 penalty_last_n      # Repetition penalty window
float32 penalty_repeat    # Repetition penalty
int32 mirostat            # Mirostat mode (0 = disabled, 1 = v1, 2 = v2)
float32 mirostat_tau      # Mirostat tau
float32 mirostat_eta      # Mirostat eta
