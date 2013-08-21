#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Elie Khoury <Elie.Khoury@idiap.ch>
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

"""Feature extraction tools"""

import bob
import numpy
import math
from .. import utils

class NullExtractor:
  """Skips proprocessing files by simply copying the contents into an hdf5 file 
  (and perform gray scale conversion if required)"""
  def __init__(self, config):
    self.m_color_channel = config.color_channel if hasattr(config, 'color_channel') else 'gray'
    
  def __call__(self, input_file, output_file, annotations = None):
    image = bob.io.load(str(input_file))
    # convert to grayscale
    image = utils.gray_channel(image, self.m_color_channel)
    image = image.astype(numpy.float64)
    bob.io.save(image, output_file)

from Cepstral import Cepstral
from BBF import BBF

