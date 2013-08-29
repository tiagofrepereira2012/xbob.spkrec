#!/usr/bin/env python

import spkrectool

preprocessor = spkrectool.preprocessing.MOD_4HZ

# Cepstral parameters
win_length_ms = 20
win_shift_ms = 10


# VAD parameters
alpha = 2
max_iterations = 10
