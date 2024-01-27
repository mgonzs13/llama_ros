int32 n_prev                64          # number of previous tokens to remember
int32 n_probs               1           # if greater than 0, output the probabilities of top n_probs tokens

bool ignore_eos             false       # ignore end of stream token and continue generating (implies --logit-bias 2-inf)
LogitBiasArray logit_bias               # logit bias for specific tokens

float32 temp                0.80        # temperature
float32 dynatemp_range      0.0         # 0.0 = disabled
float32 dynatemp_exponent   1.0         # controls how entropy maps to temperature in dynamic temperature sampler

int32 top_k                 40          # top-k sampling (0.0 = disabled)
float32 top_p               0.95        # top-p sampling (1.0 = disabled)
float32 min_p               0.05        # min-p sampling (0.0 = disabled)
float32 tfs_z               1.00        # tail free sampling, parameter z (1.0 = disabled)
float32 typical_p           1.00        # locally typical sampling, parameter p (1.0 = disabled)

int32 penalty_last_n        64          # last n tokens consider for penalize (0 = disable penalty, -1 = context size)
float32 penalty_repeat      1.10        # penalize repeat sequence of tokens (1.0 = disabled)
float32 penalty_freq        0.00        # repeat alpha frequency penalty (0.0 = disable)
float32 penalty_present     0.00        # repeat alpha presence penalty (0.0 = disabled)

int32 mirostat              0           # Mirostart sampling (0 = disabled, 1 = mirostat, 2 = mirostat 2.0)
float32 mirostat_eta        0.10        # Mirostat learning rate, parameter eta
float32 mirostat_tau        5.0         # Mirostat target entropy, parameter tau

bool penalize_nl            true        # consider newlines as a repeatable token

string samplers_sequence    "kfypmt"    # top_k, tail_free, typical_p, top_p, min_p, temp
string grammar              ""          # optional BNF-like grammar to constrain sampling
