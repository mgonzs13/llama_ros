string model                    # Tokenizer model name

string[] tokens                 # A list of tokens indexed by the token ID used by the model
float32[] scores                # If present, the score/probability of each token. If not present, all tokens are assumed to have equal probability. If present, it must have the same length and index as tokens
int32[] token_type              # The token type (1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte). If present, it must have the same length and index as tokens
string[] merges                 # If present, the merges of the tokenizer. If not present, the tokens are assumed to be atomic
string[] added_tokens           # If present, tokens that were added after training

int32 bos_token_id              # bos token of the tokenizer
int32 eos_token_id              # eos token of the tokenizer
int32 unknown_token_id          # unknown token of the tokenizer
int32 padding_token_id          # padding token of the tokenizer
int32 separator_token_id        # separator token of the tokenizer
bool add_bos_token              # Whether to add bos

string chat_template            # Chat template