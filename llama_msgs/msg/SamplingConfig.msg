int32 n_prev                        64                              # number of previous tokens to remember
int32 n_probs                       1                               # if greater than 0, output the probabilities of top n_probs tokens
int32 min_keep                      0                               # 0 = disabled, otherwise samplers should return at least min_keep tokens

bool ignore_eos                     false                           # ignore end of stream token and continue generating (implies --logit-bias 2-inf)
LogitBiasArray logit_bias                                           # logit bias for specific tokens

float32 temp                        0.80                            # temperature
float32 dynatemp_range              0.0                             # 0.0 = disabled
float32 dynatemp_exponent           1.0                             # controls how entropy maps to temperature in dynamic temperature sampler

int32 top_k                         40                              # top-k sampling (0.0 = disabled)
float32 top_p                       0.95                            # top-p sampling (1.0 = disabled)
float32 min_p                       0.05                            # min-p sampling (0.0 = disabled)
float32 xtc_probability             0.00                            # xtc sampling (0.0 = disable)
float32 xtc_threshold               0.10                            # xtc sampling threshold (> 0.5 disables XTC)
float32 typical_p                   1.00                            # locally typical sampling, parameter p (1.0 = disabled)

int32 penalty_last_n                64                              # last n tokens consider for penalize (0 = disable penalty, -1 = context size)
float32 penalty_repeat              1.00                            # penalize repeat sequence of tokens (1.0 = disabled)
float32 penalty_freq                0.00                            # repeat alpha frequency penalty (0.0 = disable)
float32 penalty_present             0.00                            # repeat alpha presence penalty (0.0 = disabled)

float32 dry_multiplier              0.0                             # DRY repetition penalty for tokens extending repetition (0.0 = disabled)
float32 dry_base                    1.75                            # multiplier * base ^ (length of sequence before token - allowed length) (0.0 = disabled)
int32 dry_allowed_length            2                               # tokens extending repetitions beyond this receive penalty
int32 dry_penalty_last_n            -1                              # how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
string[] dry_sequence_breakers      ["\\n", ":", "\\\"", "*"]       # default sequence breakers for DRY  

int32 mirostat                      0                               # Mirostart sampling (0 = disabled, 1 = mirostat, 2 = mirostat 2.0)
float32 mirostat_eta                0.10                            # Mirostat learning rate, parameter eta
float32 mirostat_tau                5.0                             # Mirostat target entropy, parameter tau

string samplers_sequence            "edkypmxt"                       # PENALTIES, DRY, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC, TEMP

string grammar                      ""                              # optional BNF-like grammar to constrain sampling
string grammar_schema               ""                              # grammar schema that defines a JSON BNF grammar
bool grammar_lazy
GrammarTrigger[] grammar_trigger_words
int32[] grammar_trigger_tokens
int32[] preserved_tokens

int32[] penalty_prompt_tokens                                       # list of tokens to penalize
bool use_penalty_prompt_tokens      false                           # whether to penalize tokens