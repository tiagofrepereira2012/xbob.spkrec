#!/usr/bin/env python

import spkrectool

preprocessor = spkrectool.preprocessing.Cepstral

# Cepstral parameters
win_length_ms = 20
win_shift_ms = 10
n_filters = 24 
dct_norm = False
#dct_norm = math.sqrt(2.0 / n_filters)
f_min = 0.
f_max = 4000
delta_win = 2
mel_scale = False
withEnergy = True
withDelta = True
withDeltaDelta = True
withDeltaEnergy = True
withDeltaDeltaEnergy = True
n_ceps = 19 # 0-->18
pre_emphasis_coef = 0.95
energy_mask = n_ceps # 19

#VAD parameters
withVADFiltering = True
useMod4Hz = False
#existingMod4HzPath = '/idiap/user/ekhoury/LOBI/work/Modulation_4Hz/banca/4Hz/'
win_shift_ms_2 = 16 # for the modulation energy
Threshold = 1.1 # for the modulation energy

useExistingVAD = False
#existingVADPath = '/idiap/temp/ekhoury/SRE2012/work/I4U/IDIAP/DATA/vad/exp/' # should be precised if useExistingVAD is equal True



# Normalization
import numpy
normalizeFeatures = True
features_mask = numpy.concatenate((numpy.arange(0,n_ceps), numpy.arange(n_ceps,60)))
#mask1 = numpy.concatenate((numpy.arange(0,n_ceps), numpy.arange(n_ceps+1,2*(n_ceps+1)))) # [0-->18, 20-->39]
#features_mask = numpy.concatenate((mask1, numpy.arange(2*(n_ceps+1),3*(n_ceps+1)-1))) #[40-->59]
#mask1 = numpy.concatenate((numpy.arange(0,16), numpy.arange(n_ceps+1,37))) # [0-->18, 20-->39]
#features_mask = numpy.concatenate((mask1, numpy.arange(2*(n_ceps+1),56))) #[40-->59]

# VAD parameters
alpha = 2
max_iterations = 10
