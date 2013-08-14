#!bin/python
# vim: set fileencoding=utf-8 :

import bob
import time, imp
import spkrectool.preprocessing
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
  m_preprocessor_config =  imp.load_source('preprocessor', "config/preprocessing/manual.py")
  preprocessor = spkrectool.preprocessing.Manual(m_preprocessor_config)
  
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
      print "Warning: file does not exist:", audio_file
  
if __name__ == "__main__":
  main()  


