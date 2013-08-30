#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:52:39 CEST 2013
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

"""Tool chain for computing verification scores"""


from UBMGMM import UBMGMMTool
from UBMGMMRegular import UBMGMMRegularTool
from ISV import ISVTool
from IVector import IVecTool

import numpy
import bob


class DummyTool:
  """This class is used to test all the possible functions of the tool chain, but it does basically nothing."""
  
  def __init__(self, setup):
    """Generates a test value that is read and written"""
    self.m_test_value = numpy.array([[1,2,3], [4,5,6], [7,8,9]], dtype = numpy.uint8)
    
  def __test__(self, file_name):
    """Simply tests that the read data is consistent"""
    test_value = bob.io.load(str(file_name))
    for y in range(3): 
      for x in range(3): 
        assert test_value[y,x] == self.m_test_value[y,x]
  
  def train_projector(self, train_files, projector_file):
    """Does not train the projector, but writes some file"""
    # try to read the training files
    for k in train_files.keys():
      bob.io.load(str(train_files[k]))
    # save something
    bob.io.save(self.m_test_value, str(projector_file))
    
  def load_projector(self, projector_file):
    """Loads the test value from file and compares it with the desired one"""
    self.__test__(projector_file)
    
  def project(self, feature):
    """Just returns the feature since this dummy implemenation does not really project the data"""
    return feature
  
  def save_feature(self, feature, feature_file):
    """Saves the given feature to the given file"""
    bob.io.save(feature, feature_file)
  
  def train_enroler(self, train_files, enroler_file):
    """Does not train the projector, but writes some file"""
    # try to read the training files
    for d in train_files.keys():
      for k in train_files[d].keys():
        bob.io.load(str(train_files[d][k]))
    # save something
    bob.io.save(self.m_test_value, str(enroler_file))
    
  def load_enroler(self, enroler_file):
    """Loads the test value from file and compares it with the desired one"""
    self.__test__(enroler_file)
    
  def enrol(self, enrol_features):
    """Returns the first feature as the model"""
    assert len(enrol_features)
    return enrol_features[0]

  def save_model(self, model, model_file):
    """Writes the model to the given model file"""
    bob.io.save(model, model_file)
    
  def read_model(self, model_file):
    """Reads the model from file"""
    return bob.io.load(model_file)
  
  def read_probe(self, probe_file):
    """Reads the probe from file"""
    return bob.io.load(probe_file)
    
  def score(self, model, probe):
    """Returns the Euclidean distance between model and probe"""
    return bob.math.euclidean_distance(model, probe)
  
