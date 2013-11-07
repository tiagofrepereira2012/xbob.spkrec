#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:42:30 CEST 2013
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

"""Cepstral Features for speaker recognition"""

import numpy
import bob
from .. import utils
import struct

class HTKFeatures:
  """Extracts Cepstral coefficents"""
  def __init__(self, config):
    self.m_config = config
    

  def normalize_features(self, params):
  #########################
  ## Initialisation part ##
  #########################
  
    normalized_vector = [ [ 0 for i in range(params.shape[1]) ] for j in range(params.shape[0]) ] 
    for index in range(params.shape[1]):
      vector = numpy.array([row[index] for row in params])
      n_samples = len(vector)
      norm_vector = utils.normalize_std_array(vector)
      
      for i in range(n_samples):
        normalized_vector[i][index]=numpy.asscalar(norm_vector[i])    
    data = numpy.array(normalized_vector)
    return data
  
 
  def HTKReader(self, input_file):
    with open(input_file, 'r') as fid:
        # The resulting array here is float32.  We could explicitly
        # cast it to double, but that will happen further up in the
        # program anyway.
        header = fid.read(12)
        (htk_size, htk_period, vec_size, htk_kind) = struct.unpack('>iihh', header)
        data = numpy.fromfile(fid, dtype='f')
        param = data.reshape((htk_size, vec_size / 4)).byteswap()
    return param

  def __call__(self, input_file, vad_file=None):
    """Read the HTK feature file and (optionally) returns normalized cepstral features for the (optionally) given VAD file"""
   
    # Read HTK features
    cepstral_features=self.HTKReader(input_file)

    # Voice activity detection
    if vad_file is None:
      labels = numpy.array(numpy.ones(cepstral_features.shape[0]), dtype=numpy.int16)
    else:
      labels=bob.io.load(str(vad_file))

    features_mask = self.m_config.features_mask
    filtered_features = numpy.ndarray(shape=((labels == 1).sum(),len(features_mask)), dtype=numpy.float64)
    i=0
    cur_i=0
   
    for row in cepstral_features:
      if labels[i]==1:
        for k in range(len(features_mask)):
          filtered_features[cur_i,k] = row[features_mask[k]]
        cur_i = cur_i + 1
      i = i+1

    if self.m_config.normalizeFeatures:
      normalized_features = self.normalize_features(filtered_features)
    else:
      normalized_features = filtered_features
    if normalized_features.shape[0] == 0:
      print("Warning: no speech found in: %s" % input_file)
      # But do not keep it empty!!! This avoids errors in next steps
      normalized_features=numpy.array([numpy.zeros(len(features_mask))])
    return normalized_features


