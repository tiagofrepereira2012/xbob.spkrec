#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
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

"""Features for speaker recognition"""

import numpy,math
import bob
import os
import time

class BBF:
  """Extracts BBF features"""
  def __init__(self, config):
    self.m_config = config
    
  def _read(self, filename):
    """Read video.FrameContainer containing preprocessed frames"""
    import pysox, tempfile, os
    fileName, fileExtension = os.path.splitext(filename)
    wav_filename = filename
    sph = False
    if fileExtension == '.sph':
      sph = True
      infile = pysox.CSoxStream(filename)
      wav_filename = tempfile.mkstemp('.wav')[1]
      outfile = pysox.CSoxStream(wav_filename,'w', infile.get_signal())
      chain = pysox.CEffectsChain(infile, outfile)
      chain.flow_effects()
      outfile.close()
    import scipy.io.wavfile
    rate, data = scipy.io.wavfile.read(str(wav_filename)); # the data is read in its native format
    if data.dtype =='int16':
      data = numpy.cast['float'](data)
    if sph: os.unlink(wav_filename)
    return [rate,data]
  
  
  numpy.set_printoptions(precision=2, threshold=numpy.nan, linewidth=200)

  
  def _normalize_std_array(self, vector):
    """Applies a unit variance normalization to an arrayset"""

    # Initializes variables
    length = 1
    n_samples = len(vector)
    mean = numpy.ndarray((length,), 'float64')
    std = numpy.ndarray((length,), 'float64')

    mean.fill(0)
    std.fill(0)

    # Computes mean and variance
    for array in vector:
      x = array.astype('float64')
      mean += x
      std += (x ** 2)

    mean /= n_samples
     
    std /= n_samples
    std -= (mean ** 2)
    std = std ** 0.5 

    arrayset = numpy.ndarray(shape=(n_samples,mean.shape[0]), dtype=numpy.float64);
    
    for i in range (0, n_samples):
      arrayset[i,:] = (vector[i]-mean) / std 
    return arrayset


  def voice_activity_detection(self, params, energy_array, base_filename):
    #########################
    ## Initialisation part ##
    #########################
    #index = self.m_config.energy_mask
    max_iterations = self.m_config.max_iterations
    alpha = self.m_config.alpha
    #features_mask = self.m_config.features_mask  
    useMod4Hz = self.m_config.useMod4Hz
    #energy_array = numpy.array([row[index] for row in params])
    n_samples = len(energy_array)
    
    normalized_energy = self._normalize_std_array(energy_array)
    
    kmeans = bob.machine.KMeansMachine(2, 1)
    m_ubm = bob.machine.GMMMachine(2, 1)
      
    kmeans_trainer = bob.trainer.KMeansTrainer()
    kmeans_trainer.convergence_threshold = 0.0005
    kmeans_trainer.max_iterations = max_iterations;
    kmeans_trainer.check_no_duplicate = True
  
    # Trains using the KMeansTrainer
    kmeans_trainer.train(kmeans, normalized_energy)
    
    [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(normalized_energy)
    means = kmeans.means
    print "means = ", means[0], means[1]
    print "variances = ", variances[0], variances[1]
    print "weights = ", weights[0], weights[1]
    
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
    print "means = ", means[0], means[1]
    print "weights = ", weights[0], weights[1]
    
    if means[0] < means[1]:
      higher = 1;
      lower = 0;
    else:
      higher = 0;
      lower = 1;
    
    if useMod4Hz:
      input_file_4hz = self.m_config.existingMod4HzPath + base_filename + '.4hz'
      Mod4Hz = self.modulation_4Hz(inputFile4Hz, n_samples, self.m_config.win_shift_ms, self.m_config.win_shift_ms_2, self.m_config.Threshold)
      normalized_Mod4Hz = self._normalize_std_array(Mod4Hz)

      X = numpy.ndarray(shape=((normalized_energy[k][0]).shape[0]+(normalized_Mod4Hz[k][0]).shape[0],len(normalized_energy)), dtype=numpy.float64)
      for k in range(len(normalized_energy)):
        X[k,:] = numpy.array([normalized_energy[k][0], normalized_Mod4Hz[k][0]])
        
      kmeans = bob.machine.KMeansMachine(3, 2)
      m_ubm = bob.machine.GMMMachine(3, 2)
      
      # Creates the KMeansTrainer
      kmeans_trainer = bob.trainer.KMeansTrainer()
      kmeans_trainer.convergence_threshold = 0.0005
      kmeans_trainer.max_iterations = max_iterations;
    
      # Trains using the KMeansTrainer
      kmeans_trainer.train(kmeans, X)
      [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(X)
      means = kmeans.means

      # Initializes the GMM
      m_ubm.means = means
      m_ubm.variances = variances
      m_ubm.weights = weights
      m_ubm.set_variance_thresholds(0.0005)
      
      trainer = bob.trainer.ML_GMMTrainer(True, True, True)
      trainer.convergence_threshold = 0.0005
      trainer.max_iterations = 25
      trainer.train(m_ubm, X)
      
      means = m_ubm.means
      weights = m_ubm.weights
      
      print "means = ", means[0], means[1], means[2]
      print "weights = ", weights[0], weights[1], weights[2]
      
      label = numpy.array(numpy.zeros(n_samples), dtype=numpy.int16);
      if (means[0][0] >0) and  (means[0][1] > 0):
        high = 0;
        low1=1;
        low2=2;
        high_mean_gauss = m_ubm.get_gaussian(high);
        low1_mean_gauss = m_ubm.get_gaussian(low1);
        low2_mean_gauss = m_ubm.get_gaussian(low2);
        for i in range(n_samples):
          if ( high_mean_gauss.log_likelihood(X[i]) > low1_mean_gauss.log_likelihood( X[i]) )  and ( high_mean_gauss.log_likelihood(X[i]) > low2_mean_gauss.log_likelihood( X[i]) ):
            label[i]=1
      print numpy.sum(label)
        
      if (means[1][0] >0) and  (means[1][1] > 0):
        high = 1;
        low1=0;
        low2=2;
        high_mean_gauss = m_ubm.get_gaussian(high);
        low1_mean_gauss = m_ubm.get_gaussian(low1);
        low2_mean_gauss = m_ubm.get_gaussian(low2);
        for i in range(n_samples):
          if ( high_mean_gauss.log_likelihood(X[i]) > low1_mean_gauss.log_likelihood( X[i]) )  and ( high_mean_gauss.log_likelihood(X[i]) > low2_mean_gauss.log_likelihood( X[i]) ):
            label[i]=1
      print numpy.sum(label)
      if (means[2][0] >0) and  (means[2][1] > 0):
        high = 2;
        low1=0;
        low2=1;
        high_mean_gauss = m_ubm.get_gaussian(high);
        low1_mean_gauss = m_ubm.get_gaussian(low1);
        low2_mean_gauss = m_ubm.get_gaussian(low2);
        for i in range(n_samples):
          if ( high_mean_gauss.log_likelihood(X[i]) > low1_mean_gauss.log_likelihood( X[i]) )  and ( high_mean_gauss.log_likelihood(X[i]) > low2_mean_gauss.log_likelihood( X[i]) ):
            label[i]=1
      print numpy.sum(label) 
      
      # if there is still no speech frames, find the maximum sum of mean and use it
      sum_mean_0 = means[0][0] + means[0][1]
      sum_mean_1 = means[1][0] + means[1][1]
      sum_mean_2 = means[2][0] + means[2][1]
      
      if(numpy.sum(label)==0):
        if (sum_mean_0 > sum_mean_1) and (sum_mean_0 > sum_mean_2):
          high = 0;
          low1 = 1;
          low2 = 2;
        else:
          if (sum_mean_1 > sum_mean_0) and (sum_mean_1 > sum_mean_0):
            high = 1;
            low1 = 0;
            low2 = 2;
          else:
            high = 2;
            low1 = 0;
            low2 = 1;
        
        high_mean_gauss = m_ubm.get_gaussian(high);
        low1_mean_gauss = m_ubm.get_gaussian(low1);
        low2_mean_gauss = m_ubm.get_gaussian(low2);
        for i in range(n_samples):
          #if normalized_energy[i]< Threshold:
          if ( high_mean_gauss.log_likelihood(X[i]) > low1_mean_gauss.log_likelihood( X[i]) )  and ( high_mean_gauss.log_likelihood(X[i]) > low2_mean_gauss.log_likelihood( X[i]) ):
            label[i]=1
      print "After (Energy+Mod-4hz) based VAD there are ", numpy.sum(label), " remaining over ", len(label)
    else:  
      label = numpy.array(numpy.ones(n_samples), dtype=numpy.int16);
    
      higher_mean_gauss = m_ubm.get_gaussian(higher);
      lower_mean_gauss = m_ubm.get_gaussian(lower);

      k=0;
      for i in range(n_samples):
        if higher_mean_gauss.log_likelihood(normalized_energy[i]) < lower_mean_gauss.log_likelihood( normalized_energy[i]):
          label[i]=0
        else:
          label[i]=label[i] * 1
      print "After Energy-based VAD there are ", numpy.sum(label), " remaining over ", len(label)
    
    out_params = numpy.ndarray(shape=((label == 1).sum(),params.shape[1]), dtype=numpy.float64)
    i=0;
    cur_i=0;
    
    print params.shape
    print out_params.shape
   
    for row in params:
      if label[i]==1:
        out_params[cur_i,:] = row
        cur_i = cur_i + 1
      i = i+1;
  
    return out_params
    
  ####################################
  ###    End of the Core Code      ###
  ####################################
  
  def normalize_features(self, params):
  #########################
  ## Initialisation part ##
  #########################
  
    normalized_vector = [ [ 0 for i in range(params.shape[1]) ] for j in range(params.shape[0]) ] ;
    for index in range(params.shape[1]):
      vector = numpy.array([row[index] for row in params])
      n_samples = len(vector)
      norm_vector = self._normalize_std_array(vector)
      
      for i in range(n_samples):
        normalized_vector[i][index]=numpy.asscalar(norm_vector[i]);    
    data = numpy.array(normalized_vector)
    return data
  
  
  def modulation_4Hz(self, inFile4Hz, n_samples, win_shift_1, win_shift_2, Threshold):
    #typically, win_shift_1 = 10ms, win_shift_2 =16ms
    f=open(inFile4Hz);
    list_1s_shift=[[float(i) for i in line.split()] for line in open(inFile4Hz)];
   
    len_list=len(list_1s_shift);
    valeur_16ms = numpy.array(numpy.zeros(len_list, dtype=numpy.float));
    
    valeur_16ms[0]=numpy.array(list_1s_shift[0]);
    for j in range(2, 63):
      valeur_16ms[j-1]=((j-1.0)/j)*valeur_16ms[j-2] +(1.0/j)*numpy.array(list_1s_shift[j-1]);
    
        
    for j in range(63, len_list-63):
      valeur_16ms[j-1]=numpy.array(numpy.mean(list_1s_shift[j-62:j]))
    
    
    valeur_16ms[len_list-1] = numpy.mean(list_1s_shift[len_list -1])
    for j in range(2, 63):
      valeur_16ms[len_list-j]=((j-1.0)/j)*valeur_16ms[len_list+1-j] +(1.0/j)*numpy.array(list_1s_shift[len_list-j]);
    
    label = numpy.array(numpy.zeros(n_samples), dtype=numpy.int16);
    
    Mod_4Hz = numpy.array(numpy.zeros(n_samples, dtype=numpy.float));
    for x in range(0, n_samples):
      y = int (win_shift_1 * x / win_shift_2);
      r =  (win_shift_1 * x) % win_shift_2;
      
      Mod_4Hz[x] = (1.0 - r) * valeur_16ms[numpy.minimum(y, len(valeur_16ms)-1)] + r * valeur_16ms[numpy.minimum(y+1, len(valeur_16ms)-1)];
              
      if Mod_4Hz[x] > Threshold:
        label[x]=1;
      else:
        label[x]=0;
    return Mod_4Hz

  def use_existing_vad(self,inArr, vad_file):
    f=open(vad_file)
    nsamples = len(inArr)
    dimensionality=inArr[0].shape[0]
    ns=0
    for line in f:
      line = line.strip()
      st_frame = float(line.split(' ')[2])
      en_frame = float(line.split(' ')[4])
      st_frame = min(int(st_frame * 100), nsamples)
      st_frame = max(st_frame, 0)
      en_frame = min(int(en_frame * 100), nsamples)
      en_frame = max(en_frame, 0)
      ns=ns+en_frame-st_frame

    outArr = numpy.ndarray(shape=(ns,dimensionality), dtype=numpy.float64)
    c=0
    for line in f:
      line = line.strip()
      st_frame = float(line.split(' ')[2])
      en_frame = float(line.split(' ')[4])
      st_frame = min(int(st_frame * 100), nsamples)
      st_frame = max(st_frame, 0)
      en_frame = min(int(en_frame * 100), nsamples)
      en_frame = max(en_frame, 0)
      for i in range(st_frame, en_frame):
        outArr[c,:]=inArr[i]
        c=c+1
    return outArr   


  def __call__(self, input_file, vad_file):
    """Computes and returns normalized cepstral features for the given input wave file"""
    
    print "Input file : ", input_file
    rate_wavsample = self._read(input_file)
    
    # Feature extraction
    
    # Set parameters
    wl = self.m_config.win_length_ms
    ws = self.m_config.win_shift_ms
    nf = self.m_config.n_filters
    nc = self.m_config.n_ceps

    f_min = self.m_config.f_min
    f_max = self.m_config.f_max
    dw = self.m_config.delta_win
    pre = self.m_config.pre_emphasis_coef


    ceps = bob.ap.Ceps(rate_wavsample[0], wl, ws, nf, nc, f_min, f_max, dw, pre)
    

    ceps.dct_norm = self.m_config.dct_norm
    ceps.mel_scale = self.m_config.mel_scale
    ceps.with_energy = self.m_config.withEnergy
    ceps.with_delta = self.m_config.withDelta
    ceps.with_delta_delta = self.m_config.withDeltaDelta
    #ceps.win_size = self.m_config.nfft
    

    spectrogram_features = ceps.spectrogram(rate_wavsample[1] )
   
    base_filename = os.path.splitext(os.path.basename(input_file))[0];
 
    # Voice activity detection
    if self.m_config.withVADFiltering:
    
      # If Using existing VAD
      if self.m_config.useExistingVAD:
        filename_without_extension = input_file.split('.')
        vad_file = filename_without_extension + '.phn'
        #vad_file = self.m_config.existingVADPath + base_filename + '.vad'
        print vad_file
        if os.path.exists(vad_file):
          filtered_features = self.use_existing_vad(spectrogram_features, vad_file)
        else:
          print "Warning: couldn't retreive the existing vad file " + vad_file + ", will try to do the VAD instead"
          energy_array = ceps.energy(rate_wavsample[1] )
          filtered_features = self.voice_activity_detection(spectrogram_features, energy_array, base_filename)
      # If not using existing VAD
      else:
        energy_array = ceps.energy(rate_wavsample[1] )
        filtered_features = self.voice_activity_detection(spectrogram_features, energy_array, base_filename)
    else:
      filtered_features = spectrogram_features

    if self.m_config.normalizeFeatures:
      normalized_features = self.normalize_features(filtered_features)
    else:
      normalized_features = filtered_features
    
    return normalized_features
    #bob.io.save(normalized_features, output_file)

