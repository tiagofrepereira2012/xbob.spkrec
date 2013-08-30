#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:43:14 CEST 2013
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

"""Energy-based voice activity detection for speaker recognition"""

import numpy
import bob
from .. import utils

import logging
logger = logging.getLogger("bob.c++")


class Energy:
  """Extracts the Energy"""
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
    
    logger_propagate = logger.propagate
    # Mute logger propagation
    if logger_propagate:
      logger.propagate = False    
    m_ubm = bob.machine.GMMMachine(2, 1)
      
    kmeans_trainer = bob.trainer.KMeansTrainer()
    kmeans_trainer.convergence_threshold = 0.0005
    kmeans_trainer.max_iterations = max_iterations
    kmeans_trainer.check_no_duplicate = True
  
    # Trains using the KMeansTrainer
    kmeans_trainer.train(kmeans, normalized_energy)
    
    
    [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(normalized_energy)
    means = kmeans.means
    if numpy.isnan(means[0]) or numpy.isnan(means[1]):
      print("Warning: skip this file")
      return numpy.array(numpy.zeros(n_samples), dtype=numpy.int16)
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
    
    # Enable logger propagation again
    if logger_propagate:
      logger.propagate = True
      
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
    print("After Energy-based VAD there are %d frames remaining over %d" %(numpy.sum(label), len(label)))
    
    return label


  
  
  def _compute_energy(self, input_file):
    """Computes and returns normalized cepstral features for the given input wave file"""
    
    print("Input wave file: %s" %input_file)
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
    """labels speech (1) and non-speech (0) parts of the given input wave file using Energy"""
    
    labels = self._compute_energy(input_file)
    
    bob.io.save(labels, output_file)
    
