#!/usr/bin/env python

import xbob.spkrec

# 1/ The tool
tool = xbob.spkrec.tools.JFATool

# 2/ GMM Training
n_gaussians = 256
iterk = 25
iterg_train = 25
end_acc = 0.0001
var_thd = 0.0001
update_weights = True
update_means = True
update_variances = True
norm_KMeans = True

# 3/ JFA Training
rv = 5
ru = 10 # The dimensionality of the subspace
relevance_factor = 4
n_iter_train = 10
n_iter_enrol = 1

# 4/ JFA Enrolment and scoring
iterg_enrol = 1
convergence_threshold = 0.0001
variance_threshold = 0.0001
relevance_factor = 4
responsibilities_threshold = 0
