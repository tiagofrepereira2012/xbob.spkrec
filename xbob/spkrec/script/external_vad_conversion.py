#!bin/python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:44:33 CEST 2013
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

import bob
import time, imp
import xbob.spkrec.preprocessing
import sys
from .. import utils
import os

def main():
  """Executes the main function"""
  input_file_list = sys.argv[1] # The input file list
  audio_dir = sys.argv[2] # The Audio directory
  vad_dir = sys.argv[3] # The VAD directory
  out_dir = sys.argv[4] # The Output directory
  
  # ensure output directory 
  utils.ensure_dir(out_dir)
  
  # Define the processor and the parameters
  m_preprocessor_config =  imp.load_source('preprocessor', "config/preprocessing/external.py")
  preprocessor = xbob.spkrec.preprocessing.External(m_preprocessor_config)
  
  infile=open(input_file_list)
  for filename in infile:
    filename = filename.strip()
    
    audio_file = str(os.path.join(audio_dir, filename) + '.sph')
    if os.path.exists(audio_file):
      out_file = str(os.path.join(out_dir, filename) + '.hdf5')
      vad_file = str(os.path.join(vad_dir, filename) + '.vad')
      # The VAD file is 5 columns text file
      # Column 1: segment number
      # Column 3: start time
      # Column 5: end time
      
      preprocessor(audio_file, out_file, vad_file)
    else:
      print("Warning: file does not exist: %s" %audio_file)
  
if __name__ == "__main__":
  main()  


