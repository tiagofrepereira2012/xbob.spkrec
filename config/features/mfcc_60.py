#!/usr/bin/env python

import xbob.spkrec
import numpy

feature_extractor = xbob.spkrec.feature_extraction.Cepstral

# Cepstral parameters
win_length_ms = 20
win_shift_ms = 10
n_filters = 24 
dct_norm = False
f_min = 0.0
f_max = 4000
delta_win = 2
mel_scale = True
withEnergy = True
withDelta = True
withDeltaDelta = True
withDeltaEnergy = True
withDeltaDeltaEnergy = True
n_ceps = 19 # 0-->18
pre_emphasis_coef = 0.95
energy_mask = n_ceps # 19
features_mask = numpy.arange(0,60)


# Normalization
normalizeFeatures = True


