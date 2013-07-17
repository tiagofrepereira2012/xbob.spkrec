#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Features for speaker recognition"""

import numpy,math
import bob
import os
import time
from .. import utils


class Energy:
  """Extracts Modulation of the Energy at 4Hz features"""
  def __init__(self, config):
    self.m_config = config

  def _voice_activity_detection(self, energy_array):
    #########################
    ## Initialisation part ##
    #########################
    #index = self.m_config.energy_mask
    max_iterations = self.m_config.max_iterations
    alpha = self.m_config.alpha
    n_samples = len(energy_array)

    normalized_energy = utils.normalize_std_array(energy_array)
    
    kmeans = bob.machine.KMeansMachine(2, 1)
    m_ubm = bob.machine.GMMMachine(2, 1)
      
    kmeans_trainer = bob.trainer.KMeansTrainer()
    kmeans_trainer.convergence_threshold = 0.0005
    kmeans_trainer.max_iterations = max_iterations
    kmeans_trainer.check_no_duplicate = True
  
    # Trains using the KMeansTrainer
    kmeans_trainer.train(kmeans, normalized_energy)
    
    [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(normalized_energy)
    means = kmeans.means
    #print "means = ", means[0], means[1]
    #print "variances = ", variances[0], variances[1]
    #print "weights = ", weights[0], weights[1]
    
    # Initializes the GMM
    m_ubm.means = means
    
    m_ubm.variances = variances
    m_ubm.weights = weights
    m_ubm.set_variance_thresholds(0.0005)
    
    trainer = bob.trainer.ML_GMMTrainer(True, True, True)
    trainer.convergence_threshold = 0.0005
    trainer.max_iterations = 25
    trainer.train(m_ubm, normalized_energy)
    means = m_ubm.means
    weights = m_ubm.weights
    #print "means = ", means[0], means[1]
    #print "weights = ", weights[0], weights[1]
    
    if means[0] < means[1]:
      higher = 1
      lower = 0
    else:
      higher = 0
      lower = 1
    
    label = numpy.array(numpy.ones(n_samples), dtype=numpy.int16)
    
    higher_mean_gauss = m_ubm.update_gaussian(higher)
    lower_mean_gauss = m_ubm.update_gaussian(lower)

    k=0
    for i in range(n_samples):
      if higher_mean_gauss.log_likelihood(normalized_energy[i]) < lower_mean_gauss.log_likelihood( normalized_energy[i]):
        label[i]=0
      else:
        label[i]=label[i] * 1
    print "After Energy-based VAD there are ", numpy.sum(label), " frames remaining over ", len(label)
    
    return label


  
  
  def _compute_energy(self, input_file):
    """Computes and returns normalized cepstral features for the given input wave file"""
    
    print "Input file : ", input_file
    rate_wavsample = utils.read(input_file)
    
    
    # Set parameters
    wl = self.m_config.win_length_ms
    ws = self.m_config.win_shift_ms
    
    e = bob.ap.Energy(rate_wavsample[0], wl, ws)
    energy_array = e(rate_wavsample[1])
    
    labels = self._voice_activity_detection(energy_array)

    labels = utils.smoothing(labels,10) # discard isolated speech less than 100ms

    
    return labels
    
  
  def __call__(self, input_file, output_file, annotations = None):
    """Computes and returns normalized cepstral features for the given input wave file"""
    
    labels = self._compute_energy(input_file)
    
    bob.io.save(labels, output_file)
    
