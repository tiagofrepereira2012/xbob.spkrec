#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:43:30 CEST 2013
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

"""Features for speaker recognition"""

import numpy
import bob
from .. import utils


class External:
  """Extracts Modulation of the Energy at 4Hz features"""
  def __init__(self, config):
    self.m_config = config

  
  def use_existing_vad(self, inArr, vad_file):
    f=open(vad_file)
    n_samples = len(inArr)
    labels = numpy.array(numpy.zeros(n_samples), dtype=numpy.int16)
    ns=0
    for line in f:
      line = line.strip()
      st_frame = float(line.split(' ')[2])
      en_frame = float(line.split(' ')[4])
      st_frame = min(int(st_frame * 100), n_samples)
      st_frame = max(st_frame, 0)
      en_frame = min(int(en_frame * 100), n_samples)
      en_frame = max(en_frame, 0)
      for i in range(st_frame, en_frame):
        labels[i]=1
    
    return labels


  
  def _conversion(self, input_file, vad_file):
    """Computes and returns normalized cepstral features for the given input wave file"""
    
    print("Input file: %s" %input_file)
    rate_wavsample = utils.read(input_file)
    
    
    # Set parameters
    wl = self.m_config.win_length_ms
    ws = self.m_config.win_shift_ms
    # The energy array is used to shape well the output vector "labels" (to avoid out of range values)
    e = bob.ap.Energy(rate_wavsample[0], wl, ws)
    energy_array = e(rate_wavsample[1])
    
    labels = self.use_existing_vad(energy_array, vad_file)
    
    return labels
    
  
  def __call__(self, input_file, output_file, vad_file):
    """Returns speech and non-speech labels"""
    
    labels = self._conversion(input_file, vad_file)
    
    bob.io.save(labels, output_file)
    
