#!/usr/bin/env python

import spkrectool

tool = spkrectool.tools.ISVTool

# 2/ GMM Training
n_gaussians = 512
iterk = 500
iterg_train = 500
end_acc = 0.0001
var_thd = 0.0001
update_weights = True
update_means = True
update_variances = True
norm_KMeans = True

# 3/ JFA Training
ru = 20
relevance_factor = 4
n_iter_train = 10
n_iter_enrol = 1

# 4/ JFA Enrolment and scoring
iterg_enrol = 1
convergence_threshold = 0.0001
variance_threshold = 0.0001
relevance_factor = 4
responsibilities_threshold = 0
