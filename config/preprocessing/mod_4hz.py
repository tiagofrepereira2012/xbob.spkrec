#!/usr/bin/env python

import xbob.spkrec

preprocessor = xbob.spkrec.preprocessing.MOD_4HZ

# Cepstral parameters
win_length_ms = 20
win_shift_ms = 10
n_filters = 40
f_min = 0.0
f_max = 4000
pre_emphasis_coef = 1.0

