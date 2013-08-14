#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>

import bob
import numpy
import logging
import facereclib.tools.UBMGMM as UBMGMM

#logger = logging.getLogger("bob.c++")
#logger.setLevel(logging.INFO)

class UBMGMMTool(UBMGMM):
  """This class is mainly based on facereclib.tools.UBMGMM"""

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
   

  def project_gmm(self, feature_array):
    """Computes GMM statistics against a UBM, given an input 2D numpy.ndarray of feature vectors"""
    return UBMGMM.project(self, feature_array)


