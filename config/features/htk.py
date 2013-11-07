#!/usr/bin/env python

import xbob.spkrec
import numpy

feature_extractor = xbob.spkrec.feature_extraction.HTKFeatures

# Cepstral coefficients Mask
features_mask = numpy.arange(0,60) 


# Normalization
normalizeFeatures = False


