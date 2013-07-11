#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury<elie.khoury@idiap.ch>

import os
import numpy
import bob

from .. import utils

class ToolChainIvector:
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
  
  
  def __read_feature__(self, feature_file, tool = None):
    """This function reads the feature from file. It uses the self.m_tool.read_feature() function, if available, otherwise it uses bob.io.load()"""
    if not tool:
      tool = self.m_tool
    #print str(feature_file)
    if hasattr(tool, 'read_feature'):
      return tool.read_feature(str(feature_file))
    else:
      return bob.io.load(str(feature_file))
      
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

      
  ####################################################
  ####### Functions related to UBM Projection ########
  ####################################################
  
  # Function 1/
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

  # Function 2/
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
          print str(feature_file)
          projected_ubm = tool.project_ubm(feature)
          # write it
          utils.ensure_dir(os.path.dirname(projected_ubm_file))
          projected_ubm.save(bob.io.HDF5File(str(projected_ubm_file), "w"))
          
  
  #######################################################
  ######### Functions related to TV projection  #########
  #######################################################
  
  # Function 1/
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
  
  # Function 2/
  def project_ivector_features(self, tool, extractor, indices = None, force=False):
    """Extract the ivectors for all files of the database"""
    self.m_tool = tool
    # load the projector file
    if hasattr(tool, 'project_ivector'):
      if hasattr(tool, 'load_projector'):
        tool.load_projector(self.m_file_selector.projector_file())
      if hasattr(tool, 'load_enroler'):
        tool.load_enroler(self.m_file_selector.enroler_file())
      feature_files = self.m_file_selector.feature_list()
      projected_ubm_files = self.m_file_selector.projected_ubm_list()
      projected_ivector_files = self.m_file_selector.projected_ivector_list()
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print "- Projection: splitting of index range %s" % str(indices)
      else:
        index_range = range(len(feature_files))
      print "project", len(index_range), "features from directory", os.path.dirname(feature_files[0]), "to directory", os.path.dirname(projected_ivector_files[0]), "using iVector Projector"
      for k in index_range:
        feature_file = feature_files[k]
        projected_ubm_file = projected_ubm_files[k]
        projected_ivector_file = projected_ivector_files[k]
        
        if not self.__check_file__(projected_ivector_file, force):
          # load feature
          #feature = bob.io.load(str(feature_file))
          feature = self.__read_feature__(feature_file, extractor)
          # load projected_ubm file
          print str(projected_ubm_file)
          projected_ubm = bob.machine.GMMStats(bob.io.HDF5File(str(projected_ubm_file)))
          # project ivector feature
          projected_ivector = tool.project_ivector(feature, projected_ubm)
          # write it
          utils.ensure_dir(os.path.dirname(projected_ivector_file))
          #print str(projected_ivector_file)
          self.__save_feature__(projected_ivector, str(projected_ivector_file))


  ###############################################
  ####### Functions related to whitening ########
  ###############################################
  
  # Function 1/
  def train_whitening_enroler(self, tool, dir_type=None, force=False):
    """Traines the model enrolment stage using the projected features"""
    self.m_tool = tool
    if hasattr(tool, 'train_whitening_enroler'):
      enroler_file = self.m_file_selector.whitening_enroler_file()
      print enroler_file
      if self.__check_file__(enroler_file, force, 1000):
        print "Enroler '%s' already exists." % enroler_file
      else:
        # training models
        train_files = self.m_file_selector.training_feature_list_by_clients(dir_type, 'train_whitening_enroler')
        # perform training
        print "Training Enroler '%s' using %d identities: " %(enroler_file, len(train_files))
        tool.train_whitening_enroler(train_files, str(enroler_file))

  # Function 2/
  def whitening_ivector(self, tool, dir_type=None, indices = None, force=False):
    """Extract the ivectors for all files of the database"""
    self.m_tool = tool
    # load the projector file
    if hasattr(tool, 'whitening_ivector'):        
      if hasattr(tool, 'load_whitening_enroler'):
        tool.load_whitening_enroler(self.m_file_selector.whitening_enroler_file())
      input_ivector_files = self.m_file_selector.projected_list(dir_type)
      whitened_ivector_files = self.m_file_selector.whitened_ivector_list()
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print "- Projection: splitting of index range %s" % str(indices)
      else:
        index_range = range(len(input_ivector_files))
      print "project", len(index_range), "i-vectors from directory", os.path.dirname(input_ivector_files[0]), "to directory", os.path.dirname(whitened_ivector_files[0]), "using Whitening Enroler"
      for k in index_range:
        ivector_file = input_ivector_files[k]
        whitened_ivector_file = whitened_ivector_files[k] 
        if not self.__check_file__(whitened_ivector_file, force): 
          ivector = self.m_tool.read_ivector(ivector_file)
          # project ivector feature
          whitened_ivector = tool.whitening_ivector(ivector)
          # write it
          utils.ensure_dir(os.path.dirname(whitened_ivector_file))
          #print str(input_ivector_file)
          self.__save_feature__(whitened_ivector, str(whitened_ivector_file))
  
  
  
  ##############################################
  ########## Function related to Lnorm #########
  ##############################################
  
  # Function 2/
  def lnorm_ivector(self, tool, dir_type=None, indices = None, force=False):
    """Extract the ivectors for all files of the database"""
    self.m_tool = tool
    # load the projector file
    if hasattr(tool, 'lnorm_ivector'):        
      input_ivector_files = self.m_file_selector.projected_list(dir_type)
      lnorm_ivector_files = self.m_file_selector.lnorm_ivector_list()
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print "- Projection: splitting of index range %s" % str(indices)
      else:
        index_range = range(len(input_ivector_files))
      print "project", len(index_range), "i-vectors from directory", os.path.dirname(input_ivector_files[0]), "to directory", os.path.dirname(lnorm_ivector_files[0])
      for k in index_range:
        ivector_file = input_ivector_files[k]
        lnorm_ivector_file = lnorm_ivector_files[k] 
        if not self.__check_file__(lnorm_ivector_file, force): 
          ivector = self.m_tool.read_ivector(ivector_file)
          # project ivector feature
          lnorm_ivector = tool.lnorm_ivector(ivector)
          # write it
          utils.ensure_dir(os.path.dirname(lnorm_ivector_file))
          #print str(input_ivector_file)
          self.__save_feature__(lnorm_ivector, str(lnorm_ivector_file))

  ##############################################
  ########## Functions related to LDA ##########
  ##############################################
  
  # Function 1/
  def lda_train_projector(self, tool, dir_type=None, force=False):
    """Traines the LDA projector stage using the projected features"""
    self.m_tool = tool
    if hasattr(tool, 'lda_train_projector'):
      lda_projector_file = self.m_file_selector.lda_projector_file()
      
      if self.__check_file__(lda_projector_file, force, 1000):
        print "Projector '%s' already exists." % lda_projector_file
      else:
        train_files = self.m_file_selector.training_feature_list_by_clients(dir_type, 'lda_train_projector')
        # perform LDA training
        print "Training LDA Projector '%s' using %d identities: " %(lda_projector_file, len(train_files))
        tool.lda_train_projector(train_files, str(lda_projector_file))
        
  # Function 2/
  def lda_project_ivector(self, tool, dir_type=None, indices = None, force=False):
    """Project the ivectors using LDA projection"""
    self.m_tool = tool
    # load the projector file
    if hasattr(tool, 'lda_project_ivector'):        
      if hasattr(tool, 'lda_load_projector'):
        tool.lda_load_projector(self.m_file_selector.lda_projector_file())
      input_ivector_files = self.m_file_selector.projected_list(dir_type)
      lda_projected_ivector_files = self.m_file_selector.lda_projected_ivector_list()
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print "- Projection: splitting of index range %s" % str(indices)
      else:
        index_range = range(len(input_ivector_files))
        print input_ivector_files
      print "project", len(index_range), "i-vectors from directory", os.path.dirname(input_ivector_files[0]), "to directory", os.path.dirname(lda_projected_ivector_files[0]), "using LDA Projector"
      for k in index_range:
        input_ivector_file = input_ivector_files[k]
        print input_ivector_file
        lda_projected_ivector_file = lda_projected_ivector_files[k]
        if not self.__check_file__(lda_projected_ivector_file, force):
          input_ivector = self.m_tool.read_ivector(input_ivector_file)
          # project input ivector feature using LDA
          lda_projected_ivector = tool.lda_project_ivector(input_ivector)
          # write it
          utils.ensure_dir(os.path.dirname(lda_projected_ivector_file))
          #print str(lda_projected_ivector_file)
          self.__save_feature__(lda_projected_ivector, str(lda_projected_ivector_file))


  ###################################################
  ################ WCCN projection ##################
  ###################################################
  
  # Function 1/
  def wccn_train_projector(self, tool, dir_type=None, force=False):
    """Traines the WCCN projector stage using the features given in dir_type"""
    self.m_tool = tool
    if hasattr(tool, 'wccn_train_projector'):
      wccn_projector_file = self.m_file_selector.wccn_projector_file()
      if self.__check_file__(wccn_projector_file, force, 1000):
        print "Projector '%s' already exists." % wccn_projector_file
      else:
        train_files = self.m_file_selector.training_feature_list_by_clients(dir_type, 'wccn_train_projector')
        # perform WCCN training
        print "Training WCCN Projector '%s' using %d identities: " %(wccn_projector_file, len(train_files))
        tool.wccn_train_projector(train_files, str(wccn_projector_file))
        
  # Function 2/
  def wccn_project_ivector(self, tool, dir_type=None, indices = None, force=False):
    """Project the ivectors using WCCN projection"""
    self.m_tool = tool
    # load the projector file
    if hasattr(tool, 'wccn_project_ivector'):        
      if hasattr(tool, 'wccn_load_projector'):
        tool.wccn_load_projector(self.m_file_selector.wccn_projector_file())
      input_ivector_files = self.m_file_selector.projected_list(dir_type)
      wccn_projected_ivector_files = self.m_file_selector.wccn_projected_ivector_list()
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print "- Projection: splitting of index range %s" % str(indices)
      else:
        index_range = range(len(input_ivector_files))
      print "project", len(index_range), "i-vectors from directory", os.path.dirname(input_ivector_files[0]), "to directory", os.path.dirname(wccn_projected_ivector_files[0]), "using WCCN Projector"
      for k in index_range:
        lda_projected_ivector_file = input_ivector_files[k]
        wccn_projected_ivector_file = wccn_projected_ivector_files[k]
        if not self.__check_file__(wccn_projected_ivector_file, force):
          lda_projected_ivector = self.m_tool.read_ivector(lda_projected_ivector_file)
          # project ivector feature using WCCN
          wccn_projected_ivector = tool.wccn_project_ivector(lda_projected_ivector)
          # write it
          utils.ensure_dir(os.path.dirname(wccn_projected_ivector_file))
          #print str(wccn_projected_ivector_file)
          self.__save_feature__(wccn_projected_ivector, str(wccn_projected_ivector_file))
          
  ###################################################
  ###### Function related to PLDA Enrollment ########
  ###################################################
  
  # Function 1/
  def train_plda_enroler(self, tool, dir_type=None, force=False):
    """Traines the PLDA model enrolment stage using the projected features"""
    self.m_tool = tool
    if hasattr(tool, 'train_plda_enroler'):
      enroler_file = self.m_file_selector.plda_enroler_file()
      
      if self.__check_file__(enroler_file, force, 1000):
        print "Enroler '%s' already exists." % enroler_file
      else:
        train_files = self.m_file_selector.training_feature_list_by_clients(dir_type, 'train_plda_enroler')
        # perform PLDA training
        print "Training PLDA Enroler '%s' using %d identities: " %(enroler_file, len(train_files))
        tool.train_plda_enroler(train_files, str(enroler_file))
  
  # Function 2/      
  def enrol_models(self, tool, extractor, compute_zt_norm, dir_type=None, indices = None, groups = ['dev', 'eval'], types = ['N','T'], force=False):
    """Enrol the models for 'dev' and 'eval' groups, for both models and T-Norm-models.
       This function by default used the projected features to compute the models.
       If you need unprojected features for the model, please define a variable with the name 
       use_unprojected_features_for_model_enrol"""
    if hasattr(tool, 'load_plda_enroler'):
      # read the model enrolment file
      tool.load_plda_enroler(self.m_file_selector.plda_enroler_file())
    self.m_tool = tool
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
          print model_file
          # Removes old file if required
          if not self.__check_file__(model_file, force):
            enrol_files = self.m_file_selector.enrol_files(model_id, group, dir_type)
            # load all files into memory
            enrol_features = []
            for k in enrol_files:
              # processes one file
              if os.path.exists(str(k)):
                feature = tool.read_ivector(str(k))
                enrol_features.append(feature)
              else:
                print "Warning: something is wrong with this file: ", str(k)
            model = tool.plda_enrol(enrol_features)
            print model
            # save the model
            model.save(bob.io.HDF5File(str(model_file), "w"))

    # T-Norm-Models
    if 'T' in types and compute_zt_norm:
      for group in groups:
        model_ids = self.m_file_selector.tmodel_ids(group)
        if indices != None: 
          model_ids = model_ids[indices[0]:indices[1]]
          print "Splitting of index range", indices, "to",
        print "enrol T-models of group", group
        for model_id in model_ids:
          # Path to the model
          model_file = self.m_file_selector.tmodel_file(model_id, group)
          #print model_file
          # Removes old file if required
          if not self.__check_file__(model_file, force):
            enrol_files = self.m_file_selector.tenrol_files(model_id, group, dir_type)
            # load all files into memory
            enrol_features = []
            for k in enrol_files:
              # processes one file
              print k
              feature = tool.read_ivector(str(k))
              enrol_features.append(feature)
            model = tool.plda_enrol(enrol_features)
            # save model
            self.__save_model__(model, model_file, tool)


  # Function 3/ 
  def __read_model__(self, model_files):
    """This function reads the model from file. Overload this function if your model is no numpy.ndarray."""
    return self.m_tool.read_model(model_files)
    
  # Function 4/  
  def __read_probe__(self, probe_file):
    #print str(probe_file)
    """This function reads the probe from file. Overload this function if your probe is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_probe'):
      return self.m_tool.read_probe(str(probe_file))
    else:
      return bob.io.load(str(probe_file))

  # Function 5/
  def __scores__(self, model, probe_files):
    """Compute simple scores for the given model"""
    scores = numpy.ndarray((1,len(probe_files)), 'float64')

    # Loops over the probes
    i = 0
    for k in probe_files:
      # read probe
      probe = self.__read_probe__(str(k))
      # compute score
      scores[0,i] = self.m_tool.plda_score(model, probe)
      i += 1
    # Returns the scores
    return scores
  
  # Function 6/  
  def __scores_preloaded__(self, model, probes):
    """Compute simple scores for the given model"""
    scores = numpy.ndarray((1,len(probes)), 'float64')
    # Loops over the probes
    i = 0
    for k in probes:
      # take pre-loaded probe
      probe = probes[k]
      # compute score
      scores[0,i] = self.m_tool.plda_score(model, probe)
      i += 1
    # Returns the scores
    return scores
    
  # Function 7/  
  def __probe_split__(self, selected_probe_objects, probes):
    sel = 0
    res = {}
    # iterate over all probe files
    for k in selected_probe_objects:
      # add probe
      res[k] = probes[k]
    # return the split database
    return res
    
  # Function 8/
  def __scores_a__(self, model_ids, group, compute_zt_norm, dir_type, force, preload_probes, scoring_type='plda'):
    """Computes A scores"""
    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      print "Preloading probe files"
      all_probe_files = self.m_file_selector.probe_files(group, dir_type)
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
        probe_files = self.m_file_selector.probe_files_for_model(model_id, group, dir_type)
        if scoring_type=='cosine':
          model=self.m_tool.read_ivectors(self.m_file_selector.model_files(model_id, group, dir_type))
        else:  
          model = self.m_tool.read_plda_model(self.m_file_selector.model_file(model_id, group))
        if preload_probes:
          # select the probe files for this model from all probes
          current_probes = self.__probe_split__(probe_files, all_probes)
          # compute A matrix
          a = self.__scores_preloaded__(model, current_probes)
        else:
          if scoring_type=='cosine':
            a = self.cosine_scores(model, probe_files)
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

  # Function 9/
  def __scores_b__(self, model_ids, group, dir_type, force, preload_probes, scoring_type='plda'):
    """Computes B scores"""
    # probe files:
    zprobe_objects = self.m_file_selector.zprobe_files(group, dir_type)
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
        if scoring_type=='cosine':
          model=self.m_tool.read_ivectors(self.m_file_selector.model_files(model_id, group, dir_type))
        else:  
          model = self.m_tool.read_plda_model(self.m_file_selector.model_file(model_id, group))
        if preload_probes:
          b = self.__scores_preloaded__(model, zprobes)
        else:
          if scoring_type=='cosine':
            b = self.cosine_scores(model, zprobe_objects)
          else:
            b = self.__scores__(model, zprobe_objects)
        bob.io.save(b, score_file)
  
  # Function 10/
  def __scores_c__(self, tmodel_ids, group, dir_type, force, preload_probes, scoring_type='plda'):
    """Computed C scores"""
    # probe files:
    probe_files = self.m_file_selector.probe_files(group, dir_type)
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
        print tmodel_id
        if scoring_type=='cosine':
          tmodel=self.m_tool.read_ivectors(self.m_file_selector.tmodel_files(tmodel_id, group, dir_type))
        else:  
          tmodel = self.m_tool.read_plda_model(self.m_file_selector.tmodel_file(tmodel_id, group))
        if preload_probes:
          c = self.__scores_preloaded__(tmodel, probes)
        else:
          if scoring_type=='cosine':
            c = self.cosine_scores(tmodel, probe_files)
          else:
            c = self.__scores__(tmodel, probe_files)
        bob.io.save(c, score_file)
      
  # Function 11/
  def __scores_d__(self, tmodel_ids, group, dir_type, force, preload_probes, scoring_type='plda'):
    # probe files:
    zprobe_objects = self.m_file_selector.zprobe_objects(group)
    zprobe_files = self.m_file_selector.zprobe_files(group, dir_type)
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
    # Loads the T-Norm models
    for tmodel_id in tmodel_ids:
      # test if the file is already there
      score_file = self.m_file_selector.d_same_value_file(tmodel_id, group)
      if self.__check_file__(score_file, force):
        print "Score file '%s' already exists." % (score_file)
      else:
        if scoring_type=='cosine':
          tmodel=self.m_tool.read_ivectors(self.m_file_selector.tmodel_files(tmodel_id, group, dir_type))
        else:
          tmodel = self.m_tool.read_plda_model(self.m_file_selector.tmodel_file(tmodel_id, group))
        if preload_probes:
          d = self.__scores_preloaded__(tmodel, zprobes)
        else:
          if scoring_type=='cosine':
            d = self.cosine_scores(tmodel, zprobe_files)
          else:
            d = self.__scores__(tmodel, zprobe_files)
        bob.io.save(d, self.m_file_selector.d_file(tmodel_id, group))
        tclient_id = [self.m_file_selector.m_config.db.get_client_id_from_model_id(tmodel_id)]
        d_same_value_tm = bob.machine.ztnorm_same_value(tclient_id, zprobe_ids)
        bob.io.save(d_same_value_tm, score_file)

  # Function 12/
  def compute_scores(self, tool, compute_zt_norm, dir_type, force = False, indices = None, groups = ['dev', 'eval'], types = ['A', 'B', 'C', 'D'], preload_probes = False, scoring_type = 'plda'):
    """Computes the scores for 'dev' and 'eval' groups"""
    if scoring_type == 'plda' and hasattr(tool, 'load_plda_enroler'):
      # read the model enrolment file
      tool.load_plda_enroler(self.m_file_selector.plda_enroler_file())
    
    # save tool for internal use
    self.m_tool = tool
    self.m_use_projected_ivector_dir = hasattr(tool, 'project_ivector')
    self.m_use_projected_ubm_dir = hasattr(tool, 'project_ubm')
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
        self.__scores_a__(model_ids_short, group, compute_zt_norm, dir_type, force, preload_probes, scoring_type)
      if compute_zt_norm:
        # compute B scores
        if 'B' in types:
          if indices != None: 
            model_ids_short = model_ids[indices[0]:indices[1]]
          else:
            model_ids_short = model_ids
          print "computing B scores"
          self.__scores_b__(model_ids_short, group, dir_type, force, preload_probes, scoring_type)
        # compute C scores
        if 'C' in types:
          if indices != None: 
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          print "computing C scores"
          self.__scores_c__(tmodel_ids_short, group, dir_type, force, preload_probes, scoring_type)
        # compute D scores
        if 'D' in types:
          if indices != None: 
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          print "computing D scores"
          self.__scores_d__(tmodel_ids_short, group, dir_type, force, preload_probes, scoring_type)
      
  # Function 13/
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
    
  # Function 14/
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

  # Function 15/
  def __scores_d_normalize__(self, tmodel_ids, group):
    # initialize D and D_same_value matrices
    d_for_all = None
    d_same_value = None
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

  # Function 16/
  def zt_norm(self, tool, groups = ['dev', 'eval']):
    """Computes ZT-Norm using the previously generated files"""
    for group in groups:
      self.m_use_projected_ivector_dir = hasattr(tool, 'project_ivector')
      self.m_use_projected_ubm_dir = hasattr(tool, 'project_ubm')
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

  # Function 17/      
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
  
  
  # Cosine Scoring/
  def cosine_scores(self, client_ivectors, probe_files):
    """Compute simple scores for the given model"""
    scores = numpy.ndarray((1,len(probe_files)), 'float64')

    # Loops over the probes
    i = 0
    for k in probe_files:
      # read probe
      probe = self.__read_probe__(str(k))
      # compute score
      scores[0,i] = self.m_tool.cosine_score(client_ivectors, probe)
      i += 1
    # Returns the scores
    return scores              
