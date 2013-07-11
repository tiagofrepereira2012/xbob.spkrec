#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Features for speaker recognition"""

import numpy,math
import bob
import os
import time

class MOD_4HZ:
  """Extracts Modulation of the Energy at 4Hz features"""
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
    rate, data = scipy.io.wavfile.read(str(wav_filename)) # the data is read in its native format
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
    arrayset = numpy.ndarray(shape=(n_samples,mean.shape[0]), dtype=numpy.float64)
    
    for i in range (0, n_samples):
      arrayset[i,:] = (vector[i]-mean) / std 
    return arrayset


  def voice_activity_detection(self, energy_array, mod_4hz):
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
    print energy_array.shape
    print energy_array
    normalized_energy = self._normalize_std_array(energy_array)
    print "mean_energy = ", numpy.mean(energy_array)
    print "var_energy =  ", numpy.var(energy_array)
    print "mean_energy = ", numpy.mean(normalized_energy)
    print "var_energy =  ", numpy.var(normalized_energy)
    print normalized_energy.shape
    print energy_array[290:310]
    print normalized_energy[290:310]
    
    #print normalized_energy
    normalized_Mod4Hz = mod_4hz # self._normalize_std_array(mod_4hz)
    
    print normalized_Mod4Hz[290:310]

    X = numpy.zeros((len(normalized_energy),2))
    print X.shape
    for k in range(len(normalized_energy)):
      X[k,:] = numpy.array([normalized_energy[k], normalized_Mod4Hz[k]])
    
    num_gauss =8 
    indicator = 1
    while indicator ==1 :
    
      kmeans = bob.machine.KMeansMachine(num_gauss, 2)
      m_ubm = bob.machine.GMMMachine(num_gauss, 2)
      
      # Creates the KMeansTrainer
      kmeans_trainer = bob.trainer.KMeansTrainer()
      kmeans_trainer.convergence_threshold = 0.0001

      kmeans_trainer.max_iterations = max_iterations
      kmeans_trainer.check_no_duplicate = True
    
      # Trains using the KMeansTrainer
      kmeans_trainer.train(kmeans, X)
      [variances, weights] = kmeans.get_variances_and_weights_for_each_cluster(X)
      means = kmeans.means
      #print means
      #print variances
      #print weights
      indicator = 0
      for index in range(num_gauss):
        if math.isnan(means[index][0]) or math.isnan(means[index][1]):
          indicator = 1
          num_gauss = num_gauss -1


    # Initializes the GMM
    m_ubm.means = means
    m_ubm.variances = variances
    m_ubm.weights = weights
    m_ubm.set_variance_thresholds(0.0001)
      
    trainer = bob.trainer.ML_GMMTrainer(True, True, True)
    trainer.convergence_threshold = 0.0001
    trainer.max_iterations = 25
    trainer.train(m_ubm, X)
      
    means = m_ubm.means
    weights = m_ubm.weights
    print "means = ",
    for index in range(num_gauss):  
      print means[index],
    print ""
    print "weights = ",
    for index in range(num_gauss):  
      print weights[index],
    print ""
      
    label = numpy.array(numpy.zeros(n_samples), dtype=numpy.int16)
    for index in range(num_gauss):
      if (means[index][0] >0.00) and  (means[index][1] > 0.0):
        high=index
        low_array = numpy.zeros((num_gauss -1,), dtype=numpy.uint64)
        index2=0
        for j in range(num_gauss):
          if j != index:
            low_array[index2]=j
            index2 = index2 +1
        print "high=", high
        print "lows=", low_array
        high_mean_gauss = m_ubm.get_gaussian(high)
      
        for i in range(n_samples):
          indicator = 1
          for index2 in range(num_gauss -1):
            
            if ( high_mean_gauss.log_likelihood(X[i]) < m_ubm.get_gaussian(low_array[index2]).log_likelihood(X[i]) or X[i][0]<0. or X[i][1] < 0.0 ):
              indicator = 0
          if indicator == 1:
            label[i]=1
        print label      
        print numpy.sum(label),  float(numpy.sum(label)) / float(len(label))
              
    if  float(numpy.sum(label)) / float(len(label)) < 0.3:
      print "TRY WITH MORE RISK..."
      for index in range(num_gauss):
        if (means[index][0] >0.10) and  (means[index][1] > 0.4):
          high=index
          low_array = numpy.zeros((num_gauss -1,), dtype=numpy.uint64)
          index2=0
          for j in range(num_gauss):
            if j != index:
              low_array[index2]=j
              index2 = index2 +1
          print "high=", high
          print "lows=", low_array
          high_mean_gauss = m_ubm.get_gaussian(high)
      
          for i in range(n_samples):
            indicator = 1
            for index2 in range(num_gauss -1):
              if ( high_mean_gauss.log_likelihood(X[i]) < m_ubm.get_gaussian(low_array[index2]).log_likelihood(X[i]) or X[i][0]<0. or X[i][1] < 0.2):
                indicator = 0
            if indicator == 1:
              label[i]=1

          print numpy.sum(label),  float(numpy.sum(label)) / float(len(label))
    
    if  float(numpy.sum(label)) / float(len(label)) < 0.15:
      print "TRY WITH MORE RISK..."
      for index in range(num_gauss):
        if (means[index][0] >0.10) and  (means[index][1] > 0.2):
          high=index
          low_array = numpy.zeros((num_gauss -1,), dtype=numpy.uint64)
          index2=0
          for j in range(num_gauss):
            if j != index:
              low_array[index2]=j
              index2 = index2 +1
          print "high=", high
          print "lows=", low_array
          high_mean_gauss = m_ubm.get_gaussian(high)
      
          for i in range(n_samples):
            indicator = 1
            for index2 in range(num_gauss -1):
              if ( high_mean_gauss.log_likelihood(X[i]) < m_ubm.get_gaussian(low_array[index2]).log_likelihood(X[i]) ):
                indicator = 0
            if indicator == 1:
              label[i]=1

          print numpy.sum(label)    ,  float(numpy.sum(label)) / float(len(label))
    print "After (Energy+Mod-4hz) based VAD there are ", numpy.sum(label), " remaining out of ", len(label)

    return label

  def voice_activity_detection_2(self, energy_array, mod_4hz):
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
    #print energy_array.shape
    #print energy_array
    
    #print "max_energy = ", numpy.max(energy_array)
    ratio_for_threshold = 5
    #print "Threshold = ", numpy.max(energy_array) - numpy.log((100./ratio_for_threshold) * (100./ratio_for_threshold))
    
    threshold = numpy.max(energy_array) - numpy.log((100./ratio_for_threshold) * (100./ratio_for_threshold))

    energy = energy_array
    
    #print mod_4hz
    
    label = numpy.array(numpy.zeros(n_samples), dtype=numpy.int16)

    for i in range(n_samples):
      if ( energy[i] > threshold and mod_4hz[i] > 0.9 ):
        label[i]=1
      
    #print label      
    #print numpy.sum(label),  float(numpy.sum(label)) / float(len(label))
              
    if  float(numpy.sum(label)) / float(len(label)) < 0.5:
      #print "TRY WITH MORE RISK 1..."
      for i in range(n_samples):
        if ( energy[i] > threshold and mod_4hz[i] > 0.5 ):
          label[i]=1

    print numpy.sum(label),  float(numpy.sum(label)) / float(len(label))
    
    
    if  float(numpy.sum(label)) / float(len(label)) < 0.5:
      #print "TRY WITH MORE RISK 2..."
      for i in range(n_samples):
        if ( energy[i] > threshold and mod_4hz[i] > 0.2 ):
          label[i]=1
    print numpy.mean(label)
    if  float(numpy.sum(label)) / float(len(label)) < 0.5: # This is especial for short segments (less than 2s)...
      #print "TRY WITH MORE RISK 3..."
      if (len(energy) < 200 ) or (numpy.sum(label) == 0) or (numpy.mean(label)<0.025):
        for i in range(n_samples):
          if ( energy[i] > threshold ):
            label[i]=1


    #print numpy.sum(label),  float(numpy.sum(label)) / float(len(label))
    
    #print "After (Energy+Mod-4hz) based VAD there are ", numpy.sum(label), " remaining out of ", len(label)

    return label 
  
  def averaging(self, list_1s_shift):

    len_list=len(list_1s_shift)
    sample_level_value = numpy.array(numpy.zeros(len_list, dtype=numpy.float))
    
    sample_level_value[0]=numpy.array(list_1s_shift[0])
    for j in range(2, numpy.min([len_list, 100])):
      sample_level_value[j-1]=((j-1.0)/j)*sample_level_value[j-2] +(1.0/j)*numpy.array(list_1s_shift[j-1])
    for j in range(numpy.min([len_list, 100]), len_list-100 +1):
      sample_level_value[j-1]=numpy.array(numpy.mean(list_1s_shift[j-100:j]))
    sample_level_value[len_list-1] = list_1s_shift[len_list -1]
    for j in range(2, numpy.min([len_list, 100]) + 1):
      sample_level_value[len_list-j]=((j-1.0)/j)*sample_level_value[len_list+1-j] +(1.0/j)*numpy.array(list_1s_shift[len_list-j])
    return sample_level_value


  
  def pass_band_filtering(self, energy_bands, fs):
    order = 2
    Nyq = float(fs/2)
    Wo = float(4/Nyq)
    #print "Wo=", Wo
    #print "fs=", fs
    #print "Nyq=" , Nyq
    #print "order = ", order
    Wn = [(Wo - 0.5/Nyq), Wo + 0.5/Nyq]
    #print Wn
    import scipy.signal
    b, a = scipy.signal.butter(order, Wn, btype='band')
    #print b
    #print a
    res = scipy.signal.lfilter(b, a, energy_bands)
    return res.T
  
    
  def modulation_4hz(self, filtering_res, rate_wavsample):
    fs = rate_wavsample[0]
    win_length = int (fs * self.m_config.win_length_ms / 1000)
    win_shift = int (fs * self.m_config.win_shift_ms / 1000)
       
    Energy = filtering_res.sum(axis=0)
    #print "Energy.shape =", Energy.shape
    #print numpy.mean(Energy)
    mean_Energy = numpy.mean(Energy)
    
    win_size = int (2.0 ** math.ceil(math.log(win_length) / math.log(2)))
    #print "win_size = ", win_size
    n_frames = 1 + (rate_wavsample[1].shape[0] - win_length) / win_shift
    range_modulation = int(fs/win_shift) # This corresponds to 1 sec 
    #print "range_modulation = ", range_modulation
    res = numpy.zeros(n_frames)
    if n_frames < range_modulation:
      return res
    for w in range(0,n_frames-range_modulation):
      E_range=Energy[w:w+range_modulation-1] # computes the modulation every 10 ms 
      if (E_range<1.).any():
        res[w] = 0
      else:
        E_range = E_range/mean_Energy 
        res[w] = numpy.var(E_range)
    res[n_frames-range_modulation:n_frames] = res[n_frames-range_modulation-1] 
    #print "max_mod_4hz = ", numpy.max(res)
    return res 
    
  def smoothing(self, labels, smoothing_window):
    print numpy.sum(labels)
    if numpy.sum(labels)< 10:
      return labels
    segments = []
    for k in range(1,len(labels)-1):
      if labels[k]==0 and labels[k-1]==1 and labels[k+1]==1 :
        labels[k]=1
    for k in range(1,len(labels)-1):
      if labels[k]==1 and labels[k-1]==0 and labels[k+1]==0 :
        labels[k]=0
    
    seg = numpy.array([0,0,labels[0]])
    for k in range(1,len(labels)):
      if labels[k] != labels[k-1]:
        seg[1]=k-1
        #print seg
        segments.append(seg)
        seg = numpy.array([k,k,labels[k]])
    seg[1]=len(labels)-1
    #print seg
    segments.append(seg)
    #print segments
    #print len(segments)
    #print segments
    #print len(segments)

    if len(segments) < 2:
      return labels
      
    curr = segments[0]
    next = segments[1]
    
    if curr[2]==1 and (curr[1]-curr[0]+1) < smoothing_window and (next[1]-next[0]+1) > smoothing_window:
      labels[curr[0] : (curr[1]+1)] = numpy.zeros(curr[1] - curr[0] + 1)
      curr[2]=0
    if curr[2]==0 and (curr[1]-curr[0]+1) < smoothing_window and (next[1]-next[0]+1) > smoothing_window:
      labels[curr[0] : (curr[1]+1)] = numpy.ones(curr[1] - curr[0] + 1)
      curr[2]=1
    if curr[2]==1 : 
      labels[curr[1]+1] = 1
    for k in range(1,len(segments)-1):
      prev = segments[k-1]
      curr = segments[k]
      next = segments[k+1]
      if curr[2]==1 and (curr[1]-curr[0]+1) < smoothing_window and (prev[1]-prev[0]+1) > smoothing_window and (next[1]-next[0]+1) > smoothing_window:
        labels[curr[0] : (curr[1]+1)] = numpy.zeros(curr[1] - curr[0] + 1)
        curr[2]=0
      if curr[2]==0 and (curr[1]-curr[0]+1) < smoothing_window and (prev[1]-prev[0]+1) > smoothing_window and (next[1]-next[0]+1) > smoothing_window:
        labels[curr[0] : (curr[1]+1)] = numpy.ones(curr[1] - curr[0] + 1)
        curr[2]=1
      if curr[2]==1 : 
        labels[curr[0]-1] = 1
        labels[curr[1]+1] = 1
    prev = segments[-2]
    curr = segments[-1]
    if curr[2]==1 and (curr[1]-curr[0]+1) < smoothing_window and (prev[1]-prev[0]+1) > smoothing_window:
      labels[curr[0] : (curr[1]+1)] = numpy.zeros(curr[1] - curr[0] + 1)
      curr[2]=0
    if curr[2]==0 and (curr[1]-curr[0]+1) < smoothing_window and (prev[1]-prev[0]+1) > smoothing_window:
      labels[curr[0] : (curr[1]+1)] = numpy.ones(curr[1] - curr[0] + 1)
      curr[2]=1
    if curr[2]==1 : 
      labels[curr[0]-1] = 1
    
    """
    segments = []    
    seg = numpy.array([0,0,labels[0]])
    for k in range(1,len(labels)):
      if labels[k] != labels[k-1]:
        seg[1]=k-1
        segments.append(seg)
        seg = numpy.array([k,k,labels[k]])
    seg[1]=len(labels)-1
    segments.append(seg)
    print segments
    print len(segments)
    """
    return labels
  
  
  def mod_4hz(self, input_file):
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

    c = bob.ap.Spectrogram(rate_wavsample[0], wl, ws, nf, f_min, f_max, pre)
    
    c.energy_filter=True
    c.log_filter=False
    c.energy_bands=True

    energy_bands = c(rate_wavsample[1])
    #bob.io.save(energy_bands, 'energy_bands_new2.hdf5')
        
    filtering_res = self.pass_band_filtering(energy_bands, rate_wavsample[0])

    mod_4hz = self.modulation_4hz(filtering_res, rate_wavsample)
    
    print "mod_4hz =", mod_4hz
    
    mod_4hz = self.averaging(mod_4hz)
    print len(mod_4hz)
    print mod_4hz

    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    e = bob.ap.Energy(rate_wavsample[0], wl, ws)
    energy_array = e(rate_wavsample[1])
    
    print energy_array
    
    labels = self.voice_activity_detection_2(energy_array, mod_4hz)
    #print labels

    labels = self.smoothing(labels,10) # discard isolated speech less than 100ms
    #print numpy.sum(labels)
    
    return labels
    
  
  def __call__(self, input_file, output_file, annotations = None):
    """Computes and returns normalized cepstral features for the given input wave file"""
    
    labels = self.mod_4hz(input_file)
    
    bob.io.save(labels, output_file)
    
