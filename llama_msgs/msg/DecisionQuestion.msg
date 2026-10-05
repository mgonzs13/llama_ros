# Question types
uint8 CHOICE = 0
uint8 SCORE  = 1
uint8 NOUL   = 2

uint8 type
string instructions
string[] keys          # CHOICE: option keys; otherwise ignored
string[] descriptions  # CHOICE: optional, aligned with keys
                       # SCORE:  ordered levels (2..10)
                       # NOUL:   optional [false, true]
