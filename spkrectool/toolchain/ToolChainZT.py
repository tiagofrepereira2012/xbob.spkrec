#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>

import os
import numpy
import bob

from .. import utils

class ToolChainZT:
  """This class includes functionalities for a default tool chain to produce verification scores"""
  
  def __init__(self, file_selector):
    """Initializes the tool chain object with the current file selector"""
    self.m_file_selector = file_selector
    
  def __save_feature__(self, data, filename):
    """Saves the given feature to the given file""" 
    utils.ensure_dir(os.path.dirname(filename))
    if hasattr(self.m_tool, 'save_feature'):
      # Tool has a save_feature function, so use this one
      self.m_tool.save_feature(data, str(filename))
    elif hasattr(data, 'save'):
      # this is some class that supports saving itself
      data.save(bob.io.HDF5File(str(filename), "w"))
    else:
      # this is most probably a numpy.ndarray that can be saved by bob.io.save
      bob.io.save(data, str(filename))
  
  def __save_scores__(self, score_file, scores, probe_objects, client_id):
    """Saves the scores into a text file."""
    assert len(probe_objects) == scores.shape[1]
    with open(score_file, 'w') as f:
      for i in range(len(probe_objects)):
        probe_object = probe_objects[i]
        f.write(str(client_id) + " " + str(probe_object.client_id) + " " + str(probe_object.path) + " " + str(scores[0,i]) + "\n")
        
    
  def __save_model__(self, data, filename, tool = None):
    utils.ensure_dir(os.path.dirname(filename))
    if tool == None:
      tool = self.m_tool
    # Tool has a save_model function, so use this one
    if hasattr(tool, 'save_model'):
      tool.save_model(data, str(filename))
    elif hasattr(data, 'save'):
      # this is some class that supports saving itself
      data.save(bob.io.HDF5File(str(filename), "w"))
    else:
      # this is most probably a numpy.ndarray that can be saved by bob.io.save
      bob.io.save(data, str(filename))
 
  def __check_file__(self, filename, force, expected_file_size = 1):
    """Checks if the file exists and has size greater or equal to expected_file_size.
    If the file is to small, or if the force option is set to true, the file is removed.
    This function returns true is the file is there, otherwise false"""
    if os.path.exists(filename):
      if force or os.path.getsize(filename) < expected_file_size:
        print "Removing old file '%s'." % filename
        os.remove(filename)
        return False
      else:
        return True
    return False
    

  def check_features(self,features):
    """ Check if there is something wrong in the features"""
  
    import math
    n_samples = features.shape[0]
    length = features.shape[1]
    mean = numpy.ndarray((length,), 'float64')
    var = numpy.ndarray((length,), 'float64')

    mean.fill(0.)
    var.fill(0.)
  
    for array in features:
      x = array.astype('float64')
      mean += x
      var += (x ** 2)
    mean /= n_samples
    var /= n_samples
    var -= (mean ** 2)
    #var = var ** 0.5 # sqrt(std)  
  
    indicator = 1
    for i in range(len(var)):
      if var[i]<0.0001 or math.isnan(var[i]) or n_samples<100:
        print var
        indicator = 0 # probably bad features

    return indicator
  
  def preprocess_images(self, preprocessor, indices=None, force=False):
    """Preprocesses the images with the given preprocessing tool"""
    # get the file lists      
    image_files = self.m_file_selector.original_image_list()
    preprocessed_image_files = self.m_file_selector.preprocessed_image_list()

    # select a subset of keys to iterate  
    #keys = sorted(image_files.keys())
    if indices != None:
      index_range = range(indices[0], indices[1])
      print "- Preprocessing: splitting of index range %s" % str(indices)
    else:
      index_range = range(len(image_files))

    print "preprocess", len(index_range), "images from directory", os.path.dirname(image_files[0]), "to directory", os.path.dirname(preprocessed_image_files[0])
    # iterate through the images and perform normalization

    # read eye files
    # - note: the resulting value of eye_files may be None
    annotation_list = self.m_file_selector.annotation_list()

    for k in index_range:
      #print k
      image_file = image_files[k]
      
      if os.path.exists(image_file):
        preprocessed_image_file = preprocessed_image_files[k]
        
        #print image_file + "    " + preprocessed_image_file
        
        """
        if os.path.isfile(preprocessed_image_file):
          featuresTmp = bob.io.load(str(preprocessed_image_file));
          if self.check_features(featuresTmp)==0: 
            print "FILE TO CHECK: something's wrong with the features: ", str(preprocessed_image_file)
        """

        if not self.__check_file__(preprocessed_image_file, force):
          annotations = None
          if annotation_list != None:
            # read eyes position file
            annotations = utils.read_annotations(annotation_list[k], self.m_file_selector.m_db_options.annotation_type)

          # call the image preprocessor
          utils.ensure_dir(os.path.dirname(preprocessed_image_file))
          preprocessed_image = preprocessor(str(image_file), str(preprocessed_image_file), annotations)
      else:
        print "WARNING: FILE DOES NOT EXIST: ", image_file


  
  
  def train_extractor(self, extractor, force = False):
    """Trains the feature extractor, if it requires training"""
    if hasattr(extractor,'train'):
      extractor_file = self.m_file_selector.extractor_file()
      if self.__check_file__(extractor_file, force, 1000):
        print "Extractor '%s' already exists." % extractor_file
      else:
        # train model
        if hasattr(extractor, 'use_training_images_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_clients('preprocessed', 'train_extractor')
          print "Training Extractor '%s' using %d identities: " %(extractor_file, len(train_files))
        else:
          train_files = self.m_file_selector.training_image_list() 
          print "Training Extractor '%s' using %d training files: " %(extractor_file, len(train_files))
        extractor.train(train_files, extractor_file)

    
  
  def extract_features(self, extractor, indices = None, force=False):
    """Extracts the features using the given extractor"""
    self.m_tool = extractor
    if hasattr(extractor, 'load'):
      extractor.load(self.m_file_selector.extractor_file())
    vad_files = self.m_file_selector.preprocessed_image_list()
    feature_files = self.m_file_selector.feature_list()
    wav_files = self.m_file_selector.original_image_list()

    # extract the features
    if indices != None:
      index_range = range(indices[0], indices[1])
      print "- Extraction: splitting of index range %s" % str(indices)
    else:
      index_range = range(len(vad_files))

    print "extract", len(index_range), "features from image directory", os.path.dirname(vad_files[0]), "to directory", os.path.dirname(feature_files[0])
    for k in index_range:
      vad_file = vad_files[k]
      feature_file = feature_files[k]
      wav_file = wav_files[k]
      print vad_file + "   " + wav_file
      
      if not self.__check_file__(feature_file, force):

        # extract feature
        feature = extractor(wav_file, vad_file)
        
        # Save feature
        self.__save_feature__(feature, str(feature_file))

      

  def train_projector(self, tool, force=False):
    """Train the feature extraction process with the preprocessed images of the world group"""
    if hasattr(tool,'train_projector'):
      projector_file = self.m_file_selector.projector_file()
      
      if self.__check_file__(projector_file, force, 1000):
        print "Projector '%s' already exists." % projector_file
      else:
        # train projector
        if hasattr(tool, 'use_training_features_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_clients('features', 'train_projector')
          print "Training Projector '%s' using %d identities: " %(projector_file, len(train_files))
        else:
          train_files = self.m_file_selector.training_feature_list() 
          print "Training Projector '%s' using %d training files: " %(projector_file, len(train_files))
  
        # perform training
        tool.train_projector(train_files, str(projector_file))
 


  def project_ubm_features(self, tool, extractor, indices = None, force=False):
    """Extract the features for all files of the database"""
    self.m_tool = tool

    # load the projector file
    if hasattr(tool, 'project_ubm'):
      if hasattr(tool, 'load_projector'):
        tool.load_projector(self.m_file_selector.projector_file())
      
      feature_files = self.m_file_selector.feature_list()
      projected_ubm_files = self.m_file_selector.projected_ubm_list()
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print "- Projection: splitting of index range %s" % str(indices)
      else:
        index_range = range(len(feature_files))

      print "project ", len(index_range), "features from directory", os.path.dirname(feature_files[0]), "to directory", os.path.dirname(projected_ubm_files[0]), "using UBM Projector"
      for k in index_range:
        feature_file = feature_files[k]
        projected_ubm_file = projected_ubm_files[k]
        
        if not self.__check_file__(projected_ubm_file, force):
          # load feature
          #feature = bob.io.load(str(feature_file))
          feature = self.__read_feature__(feature_file, extractor)
          # project feature
          if feature.shape[0]<10:
            print "Warning: Empty feature:", str(feature_file)
          else:
            projected_ubm = tool.project_ubm(feature)
            # write it
            utils.ensure_dir(os.path.dirname(projected_ubm_file))
            projected_ubm.save(bob.io.HDF5File(str(projected_ubm_file), "w"))
            #self.__save_feature__(projected_ubm, str(projected_ubm_file))
  
  def project_isv_features(self, tool, extractor, indices = None, force=False):
    """Extract the features for all files of the database"""
    self.m_tool = tool

    # load the projector file
    if hasattr(tool, 'project_isv'):
      if hasattr(tool, 'load_projector'):
        tool.load_projector(self.m_file_selector.projector_file())
        
      if hasattr(tool, 'load_enroler'):
        tool.load_enroler(self.m_file_selector.enroler_file())
      
      feature_files = self.m_file_selector.feature_list()
      projected_ubm_files = self.m_file_selector.projected_ubm_list()
      projected_isv_files = self.m_file_selector.projected_isv_list()
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print "- Projection: splitting of index range %s" % str(indices)
      else:
        index_range = range(len(feature_files))

      print "project", len(index_range), "features from directory", os.path.dirname(feature_files[0]), "to directory", os.path.dirname(projected_isv_files[0]), "using ISV Projector"
      for k in index_range:
        feature_file = feature_files[k]
        projected_ubm_file = projected_ubm_files[k]
        projected_isv_file = projected_isv_files[k]
        
        if not self.__check_file__(projected_isv_file, force):
          # load feature
          #feature = bob.io.load(str(feature_file))
          feature = self.__read_feature__(feature_file, extractor)
          
          # load projected_ubm file
          print str(projected_ubm_file)
          projected_ubm = bob.machine.GMMStats(bob.io.HDF5File(str(projected_ubm_file)))
          
          # project isv feature
          projected_isv = tool.project_isv(feature, projected_ubm)
          # write it
          utils.ensure_dir(os.path.dirname(projected_isv_file))
          #print str(projected_isv_file)
          self.__save_feature__(projected_isv, str(projected_isv_file))
  

  def __read_feature__(self, feature_file, tool = None):
    """This function reads the feature from file. It uses the self.m_tool.read_feature() function, if available, otherwise it uses bob.io.load()"""
    if not tool:
      tool = self.m_tool
    #print str(feature_file)
    if hasattr(tool, 'read_feature'):
      return tool.read_feature(str(feature_file))
    else:
      return bob.io.load(str(feature_file))
  
  def train_enroler(self, tool, force=False):
    """Traines the model enrolment stage using the projected features"""
    self.m_tool = tool
    use_projected_features = hasattr(tool, 'project_ubm') and not hasattr(tool, 'use_unprojected_features_for_model_enrol')
    if hasattr(tool, 'train_enroler'):
      enroler_file = self.m_file_selector.enroler_file()
      
      if self.__check_file__(enroler_file, force, 1000):
        print "Enroler '%s' already exists." % enroler_file
      else:
        if hasattr(tool, 'load_projector'):
          tool.load_projector(self.m_file_selector.projector_file())
        # training models
        train_files = self.m_file_selector.training_feature_list_by_clients('projected_ubm' if use_projected_features else 'features', 'train_enroler')
  
        # perform training
        print "Training Enroler '%s' using %d identities: " %(enroler_file, len(train_files))
        tool.train_enroler(train_files, str(enroler_file))



  def enrol_models(self, tool, extractor, compute_zt_norm, indices = None, groups = ['dev', 'eval'], types = ['N','T'], force=False):
    """Enrol the models for 'dev' and 'eval' groups, for both models and T-Norm-models.
       This function by default used the projected features to compute the models.
       If you need unprojected features for the model, please define a variable with the name 
       use_unprojected_features_for_model_enrol"""
    
    # read the projector file, if needed
    if hasattr(tool,'load_projector'):
      # read the feature extraction model
      tool.load_projector(self.m_file_selector.projector_file())
    if hasattr(tool, 'load_enroler'):
      # read the model enrolment file
      tool.load_enroler(self.m_file_selector.enroler_file())
    
    # use projected or unprojected features for model enrollment?
    use_projected_features = hasattr(tool, 'project_ubm') and not hasattr(tool, 'use_unprojected_features_for_model_enrol')
    # which tool to use to read the features...
    self.m_tool = tool if use_projected_features else extractor

    # Create Models
    if 'N' in types:
      for group in groups:
        model_ids = self.m_file_selector.model_ids(group)

        if indices != None: 
          model_ids = model_ids[indices[0]:indices[1]]
          print "Splitting of index range", indices, "to",
  
        print "enrol models of group", group
        for model_id in model_ids:
          # Path to the model
          model_file = self.m_file_selector.model_file(model_id, group)

          # Removes old file if required
          if not self.__check_file__(model_file, force):
            enrol_files = self.m_file_selector.enrol_files(model_id, group, use_projected_features)
            
            # load all files into memory
            enrol_features = []
            for k in enrol_files:
              # processes one file
              print k
              if os.path.exists(str(k)):
                feature = self.__read_feature__(str(k))
                enrol_features.append(feature)
              else:
                print "Warning: something is wrong with this file: ", str(k)
            
            model = tool.enrol(enrol_features)
            # save the model
            self.__save_model__(model, model_file, tool)

    # T-Norm-Models
    if 'T' in types and compute_zt_norm:
      for group in groups:
        model_ids = self.m_file_selector.tmodel_ids(group)
        print model_ids

        if indices != None: 
          model_ids = model_ids[indices[0]:indices[1]]
          print "Splitting of index range", indices, "to",
  
        print "enrol T-models of group", group
        for model_id in model_ids:
          # Path to the model
          model_file = self.m_file_selector.tmodel_file(model_id, group)
          print model_file

          # Removes old file if required
          if not self.__check_file__(model_file, force):
            enrol_files = self.m_file_selector.tenrol_files(model_id, group, use_projected_features)

            # load all files into memory
            enrol_features = []
            for k in enrol_files:
              print k
              # processes one file
              
              feature = self.__read_feature__(str(k))
              enrol_features.append(feature)
              
            model = tool.enrol(enrol_features)
            # save model
            self.__save_model__(model, model_file, tool)



  def __read_model__(self, model_file):
    """This function reads the model from file. Overload this function if your model is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_model'):
      return self.m_tool.read_model(str(model_file))
    else:
      return bob.io.load(str(model_file))
    
  def __read_probe__(self, probe_file):
    #print str(probe_file)
    """This function reads the probe from file. Overload this function if your probe is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_probe'):
      return self.m_tool.read_probe(str(probe_file))
    else:
      return bob.io.load(str(probe_file))

  def __scores__(self, model, probe_files):
    """Compute simple scores for the given model"""
    scores = numpy.ndarray((1,len(probe_files)), 'float64')

    # Loops over the probes
    i = 0
    for k in probe_files:
      # read probe
      probe = self.__read_probe__(str(k))
      # compute score
      scores[0,i] = self.m_tool.score(model, probe)
      i += 1
    # Returns the scores
    return scores
    
  def __scores_preloaded__(self, model, probes):
    """Compute simple scores for the given model"""
    scores = numpy.ndarray((1,len(probes)), 'float64')

    # Loops over the probes
    i = 0
    for k in probes:
      # take pre-loaded probe
      probe = probes[k]
      # compute score
      scores[0,i] = self.m_tool.score(model, probe)
      i += 1
    # Returns the scores
    return scores
    
    
  def __probe_split__(self, selected_probe_objects, probes):
    sel = 0
    res = {}
    # iterate over all probe files
    for k in selected_probe_objects:
      # add probe
      res[k] = probes[k]

    # return the split database
    return res
    

  def __scores_a__(self, model_ids, group, compute_zt_norm, force, preload_probes):
    """Computes A scores"""
    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      print "Preloading probe files"
      all_probe_files = self.m_file_selector.probe_files(group, self.m_use_projected_ubm_dir, self.m_use_projected_isv_dir)
      all_probes = {}
      # read all probe files into memory
      for k in all_probe_files:
        all_probes[k] = self.__read_probe__(str(all_probe_files[k][0]))
      print "Computing A matrix"

    # Computes the raw scores for each model
    for model_id in model_ids:
      # test if the file is already there
      score_file = self.m_file_selector.a_file(model_id, group) if compute_zt_norm else self.m_file_selector.no_norm_file(model_id, group)
      if self.__check_file__(score_file, force):
        print "Score file '%s' already exists." % (score_file)
      else:
        # get the probe split
        probe_objects = self.m_file_selector.probe_objects_for_model(model_id, group)
        probe_files = self.m_file_selector.probe_files_for_model(model_id, group, self.m_use_projected_ubm_dir, self.m_use_projected_isv_dir)
        model = self.__read_model__(self.m_file_selector.model_file(model_id, group))
        if preload_probes:
          # select the probe files for this model from all probes
          current_probes = self.__probe_split__(probe_files, all_probes)
          # compute A matrix
          a = self.__scores_preloaded__(model, current_probes)
        else:
          a = self.__scores__(model, probe_files)
  
        #if compute_zt_norm:
          # write A matrix only when you want to compute zt norm afterwards
        bob.io.save(a, self.m_file_selector.a_file(model_id, group))
  
        # Saves to text file
        self.__save_scores__(self.m_file_selector.no_norm_file(model_id, group), a, probe_objects, self.m_file_selector.client_id(model_id))
        """
        scores_list = utils.convertScoreToList(numpy.reshape(a,a.size), probe_objects)
        f_nonorm = open(self.m_file_selector.no_norm_file(model_id, group), 'w')
        for x in scores_list:
          f_nonorm.write(str(x[2]) + " " + str(x[1]) + " " + str(x[3]) + " " + str(x[4]) + "\n")
        f_nonorm.close()
        """

  def __scores_b__(self, model_ids, group, force, preload_probes):
    """Computes B scores"""
    # probe files:
    zprobe_objects = self.m_file_selector.zprobe_files(group, self.m_use_projected_ubm_dir, self.m_use_projected_isv_dir)
    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      print "Preloading probe files"
      zprobes = {}
      # read all probe files into memory
      for k in zprobe_objects:
        zprobes[k] = self.__read_probe__(str(zprobe_objects[k][0]))
      print "Computing B matrix"

    # Loads the models
    for model_id in model_ids:
      # test if the file is already there
      score_file = self.m_file_selector.b_file(model_id, group)
      if self.__check_file__(score_file, force):
        print "Score file '%s' already exists." % (score_file)
      else:
        model = self.__read_model__(self.m_file_selector.model_file(model_id, group))
        if preload_probes:
          b = self.__scores_preloaded__(model, zprobes)
        else:
          b = self.__scores__(model, zprobe_objects)
        bob.io.save(b, score_file)

  def __scores_c__(self, tmodel_ids, group, force, preload_probes):
    """Computed C scores"""
    # probe files:
    probe_files = self.m_file_selector.probe_files(group, self.m_use_projected_ubm_dir,self.m_use_projected_isv_dir)

    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      print "Preloading probe files"
      probes = {}
      # read all probe files into memory
      for k in probe_files:
        #print str(k)
        probes[k] = self.__read_probe__(str(k))
      print "Computing C matrix"

    # Computes the raw scores for the T-Norm model
    for tmodel_id in tmodel_ids:
      # test if the file is already there
      score_file = self.m_file_selector.c_file(tmodel_id, group)
      if self.__check_file__(score_file, force):
        print "Score file '%s' already exists." % (score_file)
      else:
        tmodel = self.__read_model__(self.m_file_selector.tmodel_file(tmodel_id, group))
        if preload_probes:
          c = self.__scores_preloaded__(tmodel, probes)
        else:
          c = self.__scores__(tmodel, probe_files)
        bob.io.save(c, score_file)
      
  def __scores_d__(self, tmodel_ids, group, force, preload_probes):
    # probe files:
    zprobe_objects = self.m_file_selector.zprobe_objects(group)
    zprobe_files = self.m_file_selector.zprobe_files(group, self.m_use_projected_ubm_dir, self.m_use_projected_isv_dir)
    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      print "Preloading probe files"
      zprobes = {}
      # read all probe files into memory
      for k in zprobe_files:
        zprobes[k] = self.__read_probe__(str(k))
      print "Computing D matrix"

    # Gets the Z-Norm impostor samples
    zprobe_ids = []
    for k in zprobe_objects:
      zprobe_ids.append(k.client_id)
      #zprobe_ids.append(zprobe_objects[k][0]) 
      
    # Loads the T-Norm models
    for tmodel_id in tmodel_ids:
      # test if the file is already there
      score_file = self.m_file_selector.d_same_value_file(tmodel_id, group)
      if self.__check_file__(score_file, force):
        print "Score file '%s' already exists." % (score_file)
      else:
        tmodel = self.__read_model__(self.m_file_selector.tmodel_file(tmodel_id, group))
        if preload_probes:
          d = self.__scores_preloaded__(tmodel, zprobes)
        else:
          d = self.__scores__(tmodel, zprobe_files)
        bob.io.save(d, self.m_file_selector.d_file(tmodel_id, group))
  
        tclient_id = [self.m_file_selector.m_config.db.get_client_id_from_tmodel_id(tmodel_id)]
        d_same_value_tm = bob.machine.ztnorm_same_value(tclient_id, zprobe_ids)
        bob.io.save(d_same_value_tm, score_file)


  def compute_scores(self, tool, compute_zt_norm, force = False, indices = None, groups = ['dev', 'eval'], types = ['A', 'B', 'C', 'D'], preload_probes = False):
    """Computes the scores for 'dev' and 'eval' groups"""
    # save tool for internal use
    self.m_tool = tool
    self.m_use_projected_isv_dir = hasattr(tool, 'project_isv')
    #print self.m_use_projected_isv_dir
    
    self.m_use_projected_ubm_dir = hasattr(tool, 'project_ubm')
    #print self.m_use_projected_ubm_dir
    
    # load the projector, if needed
    if hasattr(tool,'load_projector'):
      tool.load_projector(self.m_file_selector.projector_file())
    if hasattr(tool,'load_enroler'):
      tool.load_enroler(self.m_file_selector.enroler_file())

    for group in groups:
      print "----- computing scores for group '%s' -----" % group
      # get model ids
      model_ids = self.m_file_selector.model_ids(group)
      if compute_zt_norm:
        tmodel_ids = self.m_file_selector.tmodel_ids(group)

      # compute A scores
      if 'A' in types:
        if indices != None: 
          model_ids_short = model_ids[indices[0]:indices[1]]
        else:
          model_ids_short = model_ids
        print "computing A scores"
        self.__scores_a__(model_ids_short, group, compute_zt_norm, force, preload_probes)
      
      if compute_zt_norm:
        # compute B scores
        if 'B' in types:
          if indices != None: 
            model_ids_short = model_ids[indices[0]:indices[1]]
          else:
            model_ids_short = model_ids
          print "computing B scores"
          self.__scores_b__(model_ids_short, group, force, preload_probes)
        
        # compute C scores
        if 'C' in types:
          if indices != None: 
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          print "computing C scores"
          self.__scores_c__(tmodel_ids_short, group, force, preload_probes)
        
        # compute D scores
        if 'D' in types:
          if indices != None: 
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          print "computing D scores"
          self.__scores_d__(tmodel_ids_short, group, force, preload_probes)
      
      
  """
  def __scores_c_normalize__(self, model_ids, tmodel_ids, group):
    # read all tmodel scores
    c_for_all = None
    for tmodel_id in tmodel_ids:
      tmp = bob.io.load(self.m_file_selector.c_file(tmodel_id, group))
      if c_for_all == None:
        c_for_all = tmp
      else:
        c_for_all = numpy.vstack((c_for_all, tmp))
    # iterate over all models and generate C matrices for that specific model
    probe_objects = self.m_file_selector.probe_files(group, self.m_use_projected_ubm_dir, self.m_use_projected_isv_dir)
    for model_id in model_ids:
      # select the correct probe files for the current model
      model_probes = self.m_file_selector.probe_files_for_model(model_id, group, self.m_use_projected_ubm_dir, self.m_use_projected_isv_dir)
      probes_used = utils.probes_used_generate_vector(probe_objects, model_probes)
      c_for_model = utils.probes_used_extract_scores(c_for_all, probes_used)
      # Save C matrix to file
      bob.io.save(c_for_model, self.m_file_selector.c_file_for_model(model_id, group))
  """
  def __c_matrix_split_for_model__(self, selected_probe_objects, all_probe_objects, all_c_scores):
    """Helper function to sub-select the c-scores in case not all probe files were used to compute A scores."""
    c_scores_for_model = numpy.ndarray((all_c_scores.shape[0], len(selected_probe_objects)), numpy.float64)
    selected_index = 0
    for all_index in range(len(all_probe_objects)):
      if selected_index < len(selected_probe_objects) and selected_probe_objects[selected_index].id == all_probe_objects[all_index].id:
        c_scores_for_model[:,selected_index] = all_c_scores[:,all_index]
        selected_index += 1
    assert selected_index == len(selected_probe_objects)

    # return the split database
    return c_scores_for_model
    
  def __scores_c_normalize__(self, model_ids, t_model_ids, group):
    """Compute normalized probe scores using T-model scores."""
    # read all tmodel scores
    c_for_all = None
    for t_model_id in t_model_ids:
      tmp = bob.io.load(self.m_file_selector.c_file(t_model_id, group))
      if c_for_all == None:
        c_for_all = tmp
      else:
        c_for_all = numpy.vstack((c_for_all, tmp))
    # iterate over all models and generate C matrices for that specific model
    all_probe_objects = self.m_file_selector.probe_objects(group)
    for model_id in model_ids:
      # select the correct probe files for the current model
      probe_objects_for_model = self.m_file_selector.probe_objects_for_model(model_id, group)
      c_matrix_for_model = self.__c_matrix_split_for_model__(probe_objects_for_model, all_probe_objects, c_for_all)
      # Save C matrix to file
      bob.io.save(c_matrix_for_model, self.m_file_selector.c_file_for_model(model_id, group))

  def __scores_d_normalize__(self, tmodel_ids, group):
    # initialize D and D_same_value matrices
    d_for_all = None
    d_same_value = None
    """
    n_nonempty_tmodels = 0
    n_scores_per_model = 0
    for tmodel_id in tmodel_ids:
      tmp = bob.io.load(self.m_file_selector.d_file(tmodel_id, group))
      tmp2 = bob.io.load(self.m_file_selector.d_same_value_file(tmodel_id, group))
      if not (d_for_all == None and d_same_value == None):
        n_nonempty_tmodels += 1
        if n_scores_per_model == 0:
            n_scores_per_model = tmp.shape[1]
    d_for_all = numpy.array(shape=(n_nonempty_tmodels,n_scores_per_model), dtype=numpy.float64)
    d_same_value = numpy.array(shape=(n_nonempty_tmodels,n_scores_per_model), dtype=numpy.uint8)
    row=0
    for tmodel_id in tmodel_ids:
      tmp = bob.io.load(self.m_file_selector.d_file(tmodel_id, group))
      tmp2 = bob.io.load(self.m_file_selector.d_same_value_file(tmodel_id, group))
      if not (d_for_all == None and d_same_value == None):
        d_for_all[row,:] = tmp
        d_same_value[row,:] = tmp2
        row+=1
    """
    
    for tmodel_id in tmodel_ids:
      tmp = bob.io.load(self.m_file_selector.d_file(tmodel_id, group))
      tmp2 = bob.io.load(self.m_file_selector.d_same_value_file(tmodel_id, group))
      if d_for_all == None and d_same_value == None:
        d_for_all = tmp
        d_same_value = tmp2
      else:
        d_for_all = numpy.vstack((d_for_all, tmp))
        d_same_value = numpy.vstack((d_same_value, tmp2))

    # Saves to files
    bob.io.save(d_for_all, self.m_file_selector.d_matrix_file(group))
    bob.io.save(d_same_value, self.m_file_selector.d_same_value_matrix_file(group))
  


  def zt_norm(self, tool, groups = ['dev', 'eval']):
    """Computes ZT-Norm using the previously generated files"""
    for group in groups:
      self.m_use_projected_isv_dir = hasattr(tool, 'project_isv')
      #print self.m_use_projected_isv_dir
    
      self.m_use_projected_ubm_dir = hasattr(tool, 'project_ubm')
      #print self.m_use_projected_ubm_dir
      
      # list of models
      model_ids = self.m_file_selector.model_ids(group)
      tmodel_ids = self.m_file_selector.tmodel_ids(group)
      

      # first, normalize C and D scores
      self.__scores_c_normalize__(model_ids, tmodel_ids, group)
      # and normalize it
      self.__scores_d_normalize__(tmodel_ids, group)


      # load D matrices only once
      d = bob.io.load(self.m_file_selector.d_matrix_file(group))
      d_same_value = bob.io.load(self.m_file_selector.d_same_value_matrix_file(group)).astype(bool)
      # Loops over the model ids
      for model_id in model_ids:
        # Loads probe objects to get information about the type of access
        probe_objects = self.m_file_selector.probe_objects_for_model(model_id, group)

        # Loads A, B, C, D and D_same_value matrices
        a = bob.io.load(self.m_file_selector.a_file(model_id, group))
        b = bob.io.load(self.m_file_selector.b_file(model_id, group))
        c = bob.io.load(self.m_file_selector.c_file_for_model(model_id, group))
        
        # compute zt scores
        zt_scores = bob.machine.ztnorm(a, b, c, d, d_same_value)

        # Saves to text file
        self.__save_scores__(self.m_file_selector.zt_norm_file(model_id, group), zt_scores, probe_objects, self.m_file_selector.client_id(model_id))
        
        """
        ztscores_list = utils.convertScoreToList(numpy.reshape(ztscores_m, ztscores_m.size), probe_objects)
        sc_ztnorm_filename = self.m_file_selector.zt_norm_file(model_id, group)
        f_ztnorm = open(sc_ztnorm_filename, 'w')
        for x in ztscores_list:
          f_ztnorm.write(str(x[2]) + " " + str(x[1]) + " " + str(x[3]) + " " + str(x[4]) + "\n")
        f_ztnorm.close()
        """



  """
  def concatenate(self, compute_zt_norm, groups = ['dev', 'eval']):
    for group in groups:
      # (sorted) list of models
      model_ids = self.m_file_selector.model_ids(group)

      f = open(self.m_file_selector.no_norm_result_file(group), 'w')
      # Concatenates the scores
      for model_id in model_ids:
        model_file = self.m_file_selector.no_norm_file(model_id, group)
        assert os.path.exists(model_file)
        res_file = open(model_file, 'r')
        f.write(res_file.read())
      f.close()

      if compute_zt_norm:
        f = open(self.m_file_selector.zt_norm_result_file(group), 'w')
        # Concatenates the scores
        for model_id in model_ids:
          res_file = open(self.m_file_selector.zt_norm_file(model_id, group), 'r')
          f.write(res_file.read())
        f.close()
  """
        
  def concatenate(self, compute_zt_norm, groups = ['dev', 'eval']):
    """Concatenates all results into one (or two) score files per group."""
    for group in groups:
      print "- Scoring: concatenating score files for group '%s'" % group
      # (sorted) list of models
      model_ids = self.m_file_selector.model_ids(group)

      with open(self.m_file_selector.no_norm_result_file(group), 'w') as f:
        # Concatenates the scores
        for model_id in model_ids:
          model_file = self.m_file_selector.no_norm_file(model_id, group)
          if not os.path.exists(model_file):
            f.close()
            os.remove(self.m_file_selector.no_norm_result_file(group))
            raise IOError("The score file '%s' cannot be found. Aborting!" % model_file)

          with open(model_file, 'r') as res_file:
            f.write(res_file.read())

      if compute_zt_norm:
        with open(self.m_file_selector.zt_norm_result_file(group), 'w') as f:
          # Concatenates the scores
          for model_id in model_ids:
            model_file = self.m_file_selector.zt_norm_file(model_id, group)
            if not os.path.exists(model_file):
              f.close()
              os.remove(self.m_file_selector.zt_norm_result_file(group))
              raise IOError("The score file '%s' cannot be found. Aborting!" % model_file)

            with open(model_file, 'r') as res_file:
              f.write(res_file.read())
