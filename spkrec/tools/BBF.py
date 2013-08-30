#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:51:51 CEST 2013
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

# This is under test
import bob
import numpy
import types

class BBFModel:
  """Machine for BBF boosting"""
  def __init__(self):
    self.nsamples = 0
    self.iterations = 0
    self.feature_index1 = []
    self.feature_index2 = []
    self.beta = []
    self.dirn = []
    self.threshold = []
    self.cum_weight = []
    self.epsi = []
    self.weight = []

class BBFTool:
  """Tool chain for computing Unified Background Models and Gaussian Mixture Models of the features"""

  def __init__(self, setup):
    """Initializes the local BBF tool with the given file selector object"""
    # call base class constructor
    self.m_config = setup

  #######################################################
  ################## BBF model enrol ####################


  def save_model(self, model, model_file):
    print("Saving BBF Model")
    hdf5file = bob.io.HDF5File(model_file, "w")
    
    hdf5file.cd('/')
    hdf5file.set('nsamples', model.nsamples)
    hdf5file.set('iterations', model.iterations)
    hdf5file.set('feature_index1', model.feature_index1)
    hdf5file.set('feature_index2', model.feature_index2)
    hdf5file.set('beta', model.beta)
    hdf5file.set('dirn', model.dirn)
    hdf5file.set('threshold', model.threshold)
    hdf5file.set('cum_weight', model.cum_weight)
    hdf5file.set('epsi', model.epsi)
    hdf5file.set('weight', model.weight)

      
  def enrol(self, enrol_features, train_features):
    """Performs BBF enrolment"""
    enrol_features = enrol_features.T
    train_features = train_features.T
    ones_array = numpy.ones(enrol_features.shape[1])
    zeros_array = numpy.zeros(train_features.shape[1])
    enrol_features = numpy.vstack([enrol_features, ones_array])
    train_features = numpy.vstack([train_features, zeros_array])
    features_to_boost = numpy.hstack([enrol_features, train_features])
    print("features_to_boost.shape = %" %features_to_boost.shape)
    machine = bob.ap.Boost(min(self.m_config.max_number_samples, int(0.20*features_to_boost.shape[1])), self.m_config.boosting_iterations)
    machine(features_to_boost)
    
    model = BBFModel()
    
    print("model.get_feature_index1 = %d" %machine.get_feature_index1())
    print("model.get_feature_index2 = %d" %machine.get_feature_index2())
    print("model.get_beta = %d" %machine.get_beta())
    print("model.get_dirn = %d" %machine.get_dirn())
    print("model.get_threshold = %d" %machine.get_threshold())
    print("model.get_cum_weight = %d" %machine.get_cum_weight())
    print("model.get_epsi = %d" %machine.get_epsi())
    print("model.get_weight = %d" %machine.get_weight())
    print("model.nsamples = %d" %machine.nsamples)
    print("model.iterations = %d" %machine.iterations)
    
    model.feature_index1= machine.get_feature_index1()
    model.feature_index2= machine.get_feature_index2()
    model.beta= machine.get_beta()
    model.dirn = machine.get_dirn()
    model.threshold = machine.get_threshold()
    model.cum_weight = machine.get_cum_weight()
    model.epsi = machine.get_epsi()
    model.weight = machine.get_weight()
    model.nsamples =  machine.nsamples
    model.iterations =  machine.iterations
    
    return model

  def read_model(self, model_file):
    """Reads the projected feature to be enroled as a model"""
    hdf5file = bob.io.HDF5File(model_file)
    model = BBFModel()
    model.nsamples = hdf5file.read('nsamples')
    model.iterations = hdf5file.read('iterations')
    model.feature_index1 = hdf5file.read('feature_index1')
    model.feature_index2 = hdf5file.read('feature_index2')
    model.beta = hdf5file.read('beta')
    model.dirn = hdf5file.read('dirn')
    model.threshold = hdf5file.read('threshold')
    model.cum_weight = hdf5file.read('cum_weight')
    model.epsi = hdf5file.read('epsi')
    model.weight = hdf5file.read('weight')
    
    return model
    
  ######################################################
  ################ Feature comparison ##################

  def score(self, model, probe_feature):
    """Computes the score for the given model and the given probe using the scoring function from the config file"""
    
    probe_feature = probe_feature.T
    
    nsamples = model.nsamples
    iterations = model.iterations
    feature_index1 = model.feature_index1 
    feature_index2 = model.feature_index2
    beta = model.beta
    dirn = model.dirn
    threshold = model.threshold
    cum_weight = model.cum_weight
    epsi = model.epsi
    weight = model.weight
    
    sc0 = numpy.zeros(iterations)
    anorm = numpy.zeros(iterations)
     
    for iter in numpy.arange(iterations):
      row_feature_index1 = probe_feature[int(feature_index1[iter]-1),:]
      row_feature_index2 = probe_feature[int(feature_index2[iter]-1),:]

      row_diff = row_feature_index1 - row_feature_index2 - threshold[iter]
      
      row_sign = 2.*((row_diff > 0) - 0.5 )
      h = 0.5 + 0.5 * dirn[iter] * row_sign
      sc0[iter] = h.sum() * weight[iter]
      anorm[iter] = cum_weight[iter]
    
    sc0 = sc0 / probe_feature.shape[1]
    sc0 = numpy.cumsum(sc0) / anorm
    final_score = sc0[-1]
    return final_score

