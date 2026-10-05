string model                     # Tokenizer model name

uint32 bos_token_id              # bos token of the tokenizer
uint32 eos_token_id              # eos token of the tokenizer
uint32 unknown_token_id          # unknown token of the tokenizer
uint32 padding_token_id          # padding token of the tokenizer
uint32 separator_token_id        # separator token of the tokenizer
bool add_bos_token               # Whether to add bos

string chat_template             # Chat template

bool add_eos_token               # Whether an EOS token is added
uint32 mask_token_id             # Mask token id (0 if absent)
string[] chat_templates          # Named chat templates, e.g. ["systemone"]