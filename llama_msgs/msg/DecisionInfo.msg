bool enabled                 # Whether the evaluate_decisions service is exposed
string type                  # openjev/lev/kev/nimble/laya/clef ("" if none)
uint32 max_head_tokens       # decision max head tokens (0 if absent)
string[] temperature_names   # e.g. "choice", "choice.3_5"
float32[] temperatures       # decision temperatures
string systemone_template    # tokenizer.chat_template.systemone, "" if absent
