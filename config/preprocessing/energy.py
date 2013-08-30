#!/usr/bin/env python

import xbob.spkrec

preprocessor = xbob.spkrec.preprocessing.Energy

# Cepstral parameters
win_length_ms = 20
win_shift_ms = 10

# VAD parameters
alpha = 2
max_iterations = 10
smoothing_window = 10 # This corresponds to 100ms
