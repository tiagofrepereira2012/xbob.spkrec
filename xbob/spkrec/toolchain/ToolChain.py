#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:50:45 CEST 2013
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

import os
import numpy
import bob

from .. import utils

class ToolChain:
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
        print("Removing old file '%s'." % filename)
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
        indicator = 0 # probably bad features

    return indicator
  
  
  def select_tool_type(self, tool):
    # get the type of the tool used
    if hasattr(tool,'train_enroler'):
      if hasattr(tool,'train_plda_enroler'):
        tool_type = 'IVector'
      else:
        tool_type = 'ISV'
    else:
      tool_type = ''
    return tool_type
  
  def preprocess_audio_files(self, preprocessor, tool, indices=None, force=False):
    """Preprocesses the audio files with the given preprocessing tool"""
    # get the type of the tool used
    tool_type = self.select_tool_type(tool)
    
    # get the file lists      
    wav_files = self.m_file_selector.original_wav_list(tool_type)
    preprocessed_wav_files = self.m_file_selector.preprocessed_wav_list(tool_type)

    # select a subset of keys to iterate  
    #keys = sorted(wav_files.keys())
    if indices != None:
      index_range = range(indices[0], indices[1])
      print("- Preprocessing: splitting of index range %s" % str(indices))
    else:
      index_range = range(len(wav_files))

    print("preprocess %d wave from directory %s to directory %s" %(len(index_range), self.m_file_selector.m_config.wav_input_dir, self.m_file_selector.m_config.preprocessed_dir))
    # iterate through the audio files and perform normalization

    # read eye files
    # - note: the resulting value of eye_files may be None
    annotation_list = self.m_file_selector.annotation_list(tool_type)

    for k in index_range:
      
      wav_file = wav_files[k]
      
      if os.path.exists(wav_file):
        preprocessed_wav_file = preprocessed_wav_files[k]
        if not self.__check_file__(preprocessed_wav_file, force):
          annotations = None
          if annotation_list != None:
            # read eyes position file
            annotations = utils.read_annotations(annotation_list[k], self.m_file_selector.m_db_options.annotation_type)

          # call the wav preprocessor
          utils.ensure_dir(os.path.dirname(preprocessed_wav_file))
          preprocessed_wav = preprocessor(str(wav_file), str(preprocessed_wav_file), annotations)
      else:
        print("WARNING: FILE DOES NOT EXIST: %s" % wav_file)


  
  
  def train_extractor(self, extractor, force = False):
    """Trains the feature extractor, if it requires training"""
    if hasattr(extractor,'train'):
      extractor_file = self.m_file_selector.extractor_file()
      if self.__check_file__(extractor_file, force, 1000):
        print("Extractor '%s' already exists." % extractor_file)
      else:
        # train model
        if hasattr(extractor, 'use_training_audio_files_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_clients('preprocessed', 'train_extractor')
          print("Training Extractor '%s' using %d identities: " %(extractor_file, len(train_files)))
        else:
          train_files = self.m_file_selector.training_wav_list() 
          print("Training Extractor '%s' using %d training files: " %(extractor_file, len(train_files)))
        extractor.train(train_files, extractor_file)

    
  
  def extract_features(self, extractor, tool, indices = None, force=False):
    """Extracts the features using the given extractor"""
        # get the type of the tool used
    tool_type = self.select_tool_type(tool)
    
    self.m_tool = extractor
    if hasattr(extractor, 'load'):
      extractor.load(self.m_file_selector.extractor_file())
    vad_files = self.m_file_selector.preprocessed_wav_list(tool_type)
    feature_files = self.m_file_selector.feature_list(tool_type)
    wav_files = self.m_file_selector.original_wav_list(tool_type)

    # extract the features
    if indices != None:
      index_range = range(indices[0], indices[1])
      print("- Extraction: splitting of index range %s" % str(indices))
    else:
      index_range = range(len(vad_files))

    print("extract %d features from wav directory %s to directory %s" %(len(index_range), self.m_file_selector.m_config.wav_input_dir, self.m_file_selector.m_config.features_dir))
    for k in index_range:
      vad_file = vad_files[k]
      feature_file = feature_files[k]
      wav_file = wav_files[k]
      if not self.__check_file__(feature_file, force):
        # extract feature
        feature = extractor(wav_file, vad_file)
        if self.check_features(feature)==0: 
            print("Warning: something's wrong with the features: %s" % str(feature_file))
        # Save feature
        self.__save_feature__(feature, str(feature_file))

      
  ####################################################
  ####### Functions related to UBM Projection ########
  ####################################################
  
  # Function 1/
  def train_projector(self, tool, force=False):
    """Train the feature extraction process with the preprocessed audio files of the world group"""
    if hasattr(tool,'train_projector'):
      projector_file = self.m_file_selector.projector_file()
      
      if self.__check_file__(projector_file, force, 1000):
        print("Projector '%s' already exists." % projector_file)
      else:
        # train projector
        if hasattr(tool, 'use_training_features_sorted_by_identity'):
          train_files = self.m_file_selector.training_feature_list_by_clients('features', 'train_projector')
          print("Training Projector '%s' using %d identities: " %(projector_file, len(train_files)))
        else:
          train_files = self.m_file_selector.training_feature_list() 
          print("Training Projector '%s' using %d training files: " %(projector_file, len(train_files)))
        # perform training
        train_features = []
        for k in range(len(train_files)):
          x = bob.io.load(str(train_files[k]))
          if x.shape[0] > 0 and len(x.shape) ==2:
            train_features.append(x)
        tool.train_projector(train_features, str(projector_file))
  # Function 2/
  def project_gmm_features(self, tool, extractor, indices = None, force=False):
    """Extract the features for all files of the database"""
    self.m_tool = tool
    tool_type = self.select_tool_type(tool)
    # load the projector file
    if hasattr(tool, 'project_gmm'):
      if hasattr(tool, 'load_projector'):
        tool.load_projector(self.m_file_selector.projector_file())
      feature_files = self.m_file_selector.feature_list(tool_type)
      projected_ubm_files = self.m_file_selector.projected_ubm_list(tool_type)
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print("- Projection: splitting of index range %s" % str(indices))
      else:
        index_range = range(len(feature_files))
      print("project %d features from directory %s to directory %s using UBM Projector" %(len(index_range), self.m_file_selector.m_config.features_dir, self.m_file_selector.m_config.projected_ubm_dir))
      for k in index_range:
        feature_file = feature_files[k]
        projected_ubm_file = projected_ubm_files[k]
        
        if not self.__check_file__(projected_ubm_file, force):
          # load feature
          #feature = bob.io.load(str(feature_file))
          feature = self.__read_feature__(feature_file, extractor)
          # project feature
          projected_ubm = tool.project_gmm(feature)
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
    use_projected_features = hasattr(tool, 'project_gmm') and not hasattr(tool, 'use_unprojected_features_for_model_enrol')
    if hasattr(tool, 'train_enroler'):
      enroler_file = self.m_file_selector.enroler_file()
      if self.__check_file__(enroler_file, force, 1000):
        print("Enroler '%s' already exists." % enroler_file)
      else:
        if hasattr(tool, 'load_projector'):
          tool.load_projector(self.m_file_selector.projector_file())
        # training models
        train_files = self.m_file_selector.training_feature_list_by_clients('projected_ubm' if use_projected_features else 'features', 'train_enroler')
  
        # perform training
        print("Training Enroler '%s' using %d identities: " %(enroler_file, len(train_files)))
        tool.train_enroler(train_files, str(enroler_file))
  
 
  # Function 4/  
  def __read_probe__(self, probe_file):
    """This function reads the probe from file. Overload this function if your probe is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_probe'):
      return self.m_tool.read_probe(str(probe_file))
    else:
      return bob.io.load(str(probe_file))



    
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


  # Function 17/      
  def concatenate(self, compute_zt_norm, groups = ['dev', 'eval']):
    """Concatenates all results into one (or two) score files per group."""
    for group in groups:
      print("- Scoring: concatenating score files for group '%s'" % group)
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
  
