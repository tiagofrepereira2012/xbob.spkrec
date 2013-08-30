#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:53:40 CEST 2013
# 
# Copyright (C) 2012-2013 Idiap Research Institute, Martigny, Switzerland
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import bob
import facereclib.tools.UBMGMM as UBMGMM

class UBMGMMRegularTool (UBMGMM):
  """Tool chain for computing Universal Background Models and Gaussian Mixture Models of the features. Contrary to the class UBMGMM, the scoring is done using Regular LLR instead of its linear approximation """

  def __init__(self, setup):
    """Initializes the local UBM-GMM tool chain with the given file selector object"""
    self.m_config = setup
    self.m_ubm = None
    UBMGMM.__init__(self, number_of_gaussians=self.m_config.n_gaussians) 

    if hasattr(self.m_config, 'scoring_function'):
      self.m_scoring_function = self.m_config.scoring_function
    
    self.m_normalize_before_k_means = self.m_config.norm_KMeans
    #self.m_gaussians = self.m_config.n_gaussians
    self.m_training_threshold = self.m_config.convergence_threshold
    self.m_gmm_training_iterations = self.m_config.iterk
    self.m_variance_threshold = self.m_config.variance_threshold
    self.m_update_means = self.m_config.update_means
    self.m_update_variances = self.m_config.update_variances
    self.m_update_weights = self.m_config.update_weights
    self.m_responsibility_threshold = self.m_config.responsibilities_threshold
    self.m_relevance_factor = self.m_config.relevance_factor 
    self.m_gmm_enroll_iterations = self.m_config.iterg_enrol
    
    self.use_unprojected_features_for_model_enrol = True
 
  def read_probe(self, probe_file):
    return bob.io.load(probe_file)
  
  def score(self, model, probe):
    """Computes the score for the given model and the given probe.
       The score are Log-Likelihood.
       Therefore, the log of the likelihood ratio is obtained by computing the following difference."""

    
    score = 0
    for i in range(probe.shape[0]):
      score += model.forward(probe[i,:]) - self.m_ubm.forward(probe[i,:])
    return score/probe.shape[0]
   
