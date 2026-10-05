bool success
string error
uint8 type
string choice          # CHOICE: best key
float32 score          # SCORE: expected level
float32 noul           # NOUL: P(true)
float32 confidence
string[] keys          # ordered option keys, aligned with probabilities
float32[] probabilities
