
bool ignore_eos             false   # ignore end of stream token and continue generating (implies --logit-bias 2-inf)
LogitBiasArray logit_bias           # logit bias for specific tokens

float32 temp                0.80    # temperature

int32 top_k                 40      # top-k sampling (0.0 = disabled)
float32 top_p               0.95    # top-p sampling (1.0 = disabled)
float32 tfs_z               1.00    # tail free sampling, parameter z (1.0 = disabled)
float32 typical_p           1.00    # locally typical sampling, parameter p (1.0 = disabled)

int32 repeat_last_n         64      # last n tokens consider for penalize (0 = disable penalty, -1 = context size)
float32 repeat_penalty      1.10    # penalize repeat sequence of tokens (1.0 = disabled)
float32 presence_penalty    0.00    # repeat alpha presence penalty (0.0 = disabled)
float32 frequency_penalty   0.00    # repeat alpha frequency penalty (0.0 = disable)

int32 mirostat              0       # Mirostart sampling (0 = disabled, 1 = mirostat, 2 = mirostat 2.0)
float32 mirostat_eta        0.10    # Mirostat learning rate, parameter eta
float32 mirostat_tau        5.0     # Mirostat target entropy, parameter tau

bool penalize_nl            true    # consider newlines as a repeatable token
int32 n_probs               1       # if greater than 0, output the probabilities of top n_probs tokens

string grammar              ""      # optional BNF-like grammar to constrain sampling