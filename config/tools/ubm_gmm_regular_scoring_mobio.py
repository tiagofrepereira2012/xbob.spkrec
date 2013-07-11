#!/usr/bin/env python

import spkrectool
import bob

tool = spkrectool.tools.UBMGMMRegularTool


# 2/ GMM Training
n_gaussians = 512
iterk = 25
iterg_train = 25
update_weights = True
update_means = True
update_variances = True
norm_KMeans = True


# 3/ GMM Enrolment and scoring
iterg_enrol = 5
convergence_threshold = 0.00001
variance_threshold = 0.0001
relevance_factor = 16
responsibilities_threshold = 0

