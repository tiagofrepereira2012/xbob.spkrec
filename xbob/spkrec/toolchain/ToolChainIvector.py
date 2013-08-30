#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:50:25 CEST 2013
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
#

import os
import numpy
import bob

from .. import utils
from . import ToolChain

class ToolChainIvector(ToolChain):
  """This class includes functionalities for an I-Vector tool chain to produce verification scores"""
  
 
  # Function 2/
  def project_ivector_features(self, tool, extractor, indices = None, force=False):
    """Extract the ivectors for all files of the database"""
    self.m_tool = tool
    tool_type = self.select_tool_type(tool)
    # load the projector file
    if hasattr(tool, 'project_ivector'):
      if hasattr(tool, 'load_projector'):
        tool.load_projector(self.m_file_selector.projector_file())
      if hasattr(tool, 'load_enroler'):
        tool.load_enroler(self.m_file_selector.enroler_file())
      feature_files = self.m_file_selector.feature_list(tool_type)
      projected_ubm_files = self.m_file_selector.projected_ubm_list(tool_type)
      projected_ivector_files = self.m_file_selector.projected_ivector_list(tool_type)
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print("- Projection: splitting of index range %s" % str(indices))
      else:
        index_range = range(len(feature_files))
      print("project %d features from directory %s to directory %s" % (len(index_range), self.m_file_selector.m_config.projected_ubm_dir, self.m_file_selector.m_config.projected_ivector_dir))
      for k in index_range:
        feature_file = feature_files[k]
        projected_ubm_file = projected_ubm_files[k]
        projected_ivector_file = projected_ivector_files[k]
        
        if not self.__check_file__(projected_ivector_file, force):
          # load feature
          #feature = bob.io.load(str(feature_file))
          feature = self.__read_feature__(feature_file, extractor)
          # load projected_ubm file
          projected_ubm = bob.machine.GMMStats(bob.io.HDF5File(str(projected_ubm_file)))
          # project ivector feature
          projected_ivector = tool.project_ivector(feature, projected_ubm)
          # write it
          utils.ensure_dir(os.path.dirname(projected_ivector_file))
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
      if self.__check_file__(enroler_file, force, 1000):
        print("Enroler '%s' already exists." % enroler_file)
      else:
        # training models
        train_files = self.m_file_selector.training_feature_list_by_clients(dir_type, 'train_whitening_enroler')
        # perform training
        print("Training Enroler '%s' using %d identities: " %(enroler_file, len(train_files)))
        tool.train_whitening_enroler(train_files, str(enroler_file))

  # Function 2/
  def whitening_ivector(self, tool, dir_type=None, indices = None, force=False):
    """Extract the ivectors for all files of the database"""
    self.m_tool = tool
    tool_type = self.select_tool_type(tool)
    # load the projector file
    if hasattr(tool, 'whitening_ivector'):        
      if hasattr(tool, 'load_whitening_enroler'):
        tool.load_whitening_enroler(self.m_file_selector.whitening_enroler_file())
      input_ivector_files = self.m_file_selector.projected_list(dir_type, tool_type)
      whitened_ivector_files = self.m_file_selector.whitened_ivector_list(tool_type)
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print("- Projection: splitting of index range %s" % str(indices))
      else:
        index_range = range(len(input_ivector_files))
      print("project %d %s i-vectors to directory %s using Whitening Enroler" %(len(index_range), dir_type, self.m_file_selector.m_config.whitened_ivector_dir))
      for k in index_range:
        ivector_file = input_ivector_files[k]
        whitened_ivector_file = whitened_ivector_files[k] 
        if not self.__check_file__(whitened_ivector_file, force): 
          ivector = self.m_tool.read_ivector(ivector_file)
          # project ivector feature
          whitened_ivector = tool.whitening_ivector(ivector)
          # write it
          utils.ensure_dir(os.path.dirname(whitened_ivector_file))
          self.__save_feature__(whitened_ivector, str(whitened_ivector_file))
  
  
  
  ##############################################
  ########## Function related to Lnorm #########
  ##############################################
  
  # Function 2/
  def lnorm_ivector(self, tool, dir_type=None, indices = None, force=False):
    """Extract the ivectors for all files of the database"""
    self.m_tool = tool
    tool_type = self.select_tool_type(tool)
    # load the projector file
    if hasattr(tool, 'lnorm_ivector'):        
      input_ivector_files = self.m_file_selector.projected_list(dir_type, tool_type)
      lnorm_ivector_files = self.m_file_selector.lnorm_ivector_list(tool_type)
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print("- Projection: splitting of index range %s" % str(indices))
      else:
        index_range = range(len(input_ivector_files))
      print("project %d %s i-vectors to directory %s" %(len(index_range), dir_type, self.m_file_selector.m_config.lnorm_ivector_dir))
      for k in index_range:
        ivector_file = input_ivector_files[k]
        lnorm_ivector_file = lnorm_ivector_files[k] 
        if not self.__check_file__(lnorm_ivector_file, force): 
          ivector = self.m_tool.read_ivector(ivector_file)
          # project ivector feature
          lnorm_ivector = tool.lnorm_ivector(ivector)
          # write it
          utils.ensure_dir(os.path.dirname(lnorm_ivector_file))
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
        print("Projector '%s' already exists." % lda_projector_file)
      else:
        train_files = self.m_file_selector.training_feature_list_by_clients(dir_type, 'lda_train_projector')
        # perform LDA training
        print("Training LDA Projector '%s' using %d identities: " %(lda_projector_file, len(train_files)))
        tool.lda_train_projector(train_files, str(lda_projector_file))
        
  # Function 2/
  def lda_project_ivector(self, tool, dir_type=None, indices = None, force=False):
    """Project the ivectors using LDA projection"""
    self.m_tool = tool
    tool_type = self.select_tool_type(tool)
    # load the projector file
    if hasattr(tool, 'lda_project_ivector'):        
      if hasattr(tool, 'lda_load_projector'):
        tool.lda_load_projector(self.m_file_selector.lda_projector_file())
      input_ivector_files = self.m_file_selector.projected_list(dir_type, tool_type)
      lda_projected_ivector_files = self.m_file_selector.lda_projected_ivector_list(tool_type)
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print("- Projection: splitting of index range %s" % str(indices))
      else:
        index_range = range(len(input_ivector_files))
      print("project %d %s to directory %s using LDA Projector" %(len(index_range), dir_type, self.m_file_selector.m_config.lda_projected_ivector_dir))
      for k in index_range:
        input_ivector_file = input_ivector_files[k]
        lda_projected_ivector_file = lda_projected_ivector_files[k]
        if not self.__check_file__(lda_projected_ivector_file, force):
          input_ivector = self.m_tool.read_ivector(input_ivector_file)
          # project input ivector feature using LDA
          lda_projected_ivector = tool.lda_project_ivector(input_ivector)
          # write it
          utils.ensure_dir(os.path.dirname(lda_projected_ivector_file))
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
        print("Projector '%s' already exists." % wccn_projector_file)
      else:
        train_files = self.m_file_selector.training_feature_list_by_clients(dir_type, 'wccn_train_projector')
        # perform WCCN training
        print("Training WCCN Projector '%s' using %d identities: " %(wccn_projector_file, len(train_files)))
        tool.wccn_train_projector(train_files, str(wccn_projector_file))
        
  # Function 2/
  def wccn_project_ivector(self, tool, dir_type=None, indices = None, force=False):
    """Project the ivectors using WCCN projection"""
    self.m_tool = tool
    tool_type = self.select_tool_type(tool)
    # load the projector file
    if hasattr(tool, 'wccn_project_ivector'):        
      if hasattr(tool, 'wccn_load_projector'):
        tool.wccn_load_projector(self.m_file_selector.wccn_projector_file())
      input_ivector_files = self.m_file_selector.projected_list(dir_type, tool_type)
      wccn_projected_ivector_files = self.m_file_selector.wccn_projected_ivector_list(tool_type)
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print("- Projection: splitting of index range %s" % str(indices))
      else:
        index_range = range(len(input_ivector_files))
      print("project %d %s i-vectors to directory %s using WCCN Projector" %(len(index_range), dir_type, self.m_file_selector.m_config.wccn_projected_ivector_dir))
      for k in index_range:
        lda_projected_ivector_file = input_ivector_files[k]
        wccn_projected_ivector_file = wccn_projected_ivector_files[k]
        if not self.__check_file__(wccn_projected_ivector_file, force):
          lda_projected_ivector = self.m_tool.read_ivector(lda_projected_ivector_file)
          # project ivector feature using WCCN
          wccn_projected_ivector = tool.wccn_project_ivector(lda_projected_ivector)
          # write it
          utils.ensure_dir(os.path.dirname(wccn_projected_ivector_file))
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
        print("Enroler '%s' already exists." % enroler_file)
      else:
        train_files = self.m_file_selector.training_feature_list_by_clients(dir_type, 'train_plda_enroler')
        # perform PLDA training
        print("Training PLDA Enroler '%s' using %d identities: " %(enroler_file, len(train_files)))
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
          print("Splitting of index range %d to" %indices),
        print("enrol models of group %s" %group)
        for model_id in model_ids:
          # Path to the model
          model_file = self.m_file_selector.model_file(model_id, group)
          print("model: %s" %model_file)
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
                print("Warning: something is wrong with this file: %s" %str(k))
            model = tool.plda_enrol(enrol_features)
            # save the model
            model.save(bob.io.HDF5File(str(model_file), "w"))

    # T-Norm-Models
    if 'T' in types and compute_zt_norm:
      for group in groups:
        model_ids = self.m_file_selector.tmodel_ids(group)
        if indices != None: 
          model_ids = model_ids[indices[0]:indices[1]]
          print("Splitting of index range %d to" %indices),
        print("enrol T-models of group %s" %group)
        for model_id in model_ids:
          # Path to the model
          model_file = self.m_file_selector.tmodel_file(model_id, group)
          # Removes old file if required
          if not self.__check_file__(model_file, force):
            enrol_files = self.m_file_selector.tenrol_files(model_id, group, dir_type)
            # load all files into memory
            enrol_features = []
            for k in enrol_files:
              # processes one file
              feature = tool.read_ivector(str(k))
              enrol_features.append(feature)
            model = tool.plda_enrol(enrol_features)
            # save model
            self.__save_model__(model, model_file, tool)


  # Function 3/ 
  def __read_model__(self, model_files):
    """This function reads the model from file. Overload this function if your model is no numpy.ndarray."""
    return self.m_tool.read_model(model_files)

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
    

    
  # Function 8/
  def __scores_a__(self, model_ids, group, compute_zt_norm, dir_type, force, preload_probes, scoring_type='plda'):
    """Computes A scores"""
    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      print("Preloading probe files")
      all_probe_files = self.m_file_selector.probe_files(group, dir_type)
      all_probes = {}
      # read all probe files into memory
      for k in all_probe_files:
        all_probes[k] = self.__read_probe__(str(all_probe_files[k][0]))
      print("Computing A matrix")
    # Computes the raw scores for each model
    for model_id in model_ids:
      # test if the file is already there
      score_file = self.m_file_selector.a_file(model_id, group) if compute_zt_norm else self.m_file_selector.no_norm_file(model_id, group)
      if self.__check_file__(score_file, force):
        print("Score file '%s' already exists." % (score_file))
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


  # Function 9/
  def __scores_b__(self, model_ids, group, dir_type, force, preload_probes, scoring_type='plda'):
    """Computes B scores"""
    # probe files:
    zprobe_objects = self.m_file_selector.zprobe_files(group, dir_type)
    # preload the probe files for a faster access (and fewer network load)
    if preload_probes:
      print("Preloading probe files")
      zprobes = {}
      # read all probe files into memory
      for k in zprobe_objects:
        zprobes[k] = self.__read_probe__(str(zprobe_objects[k][0]))
      print("Computing B matrix")
    # Loads the models
    for model_id in model_ids:
      # test if the file is already there
      score_file = self.m_file_selector.b_file(model_id, group)
      if self.__check_file__(score_file, force):
        print("Score file '%s' already exists." % (score_file))
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
      print("Preloading probe files")
      probes = {}
      # read all probe files into memory
      for k in probe_files:
        probes[k] = self.__read_probe__(str(k))
      print("Computing C matrix")
    # Computes the raw scores for the T-Norm model
    for tmodel_id in tmodel_ids:
      # test if the file is already there
      score_file = self.m_file_selector.c_file(tmodel_id, group)
      if self.__check_file__(score_file, force):
        print("Score file '%s' already exists." % (score_file))
      else:
        print("T-model: %s" %tmodel_id)
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
      print("Preloading probe files")
      zprobes = {}
      # read all probe files into memory
      for k in zprobe_files:
        zprobes[k] = self.__read_probe__(str(k))
      print("Computing D matrix")
    # Gets the Z-Norm impostor samples
    zprobe_ids = []
    for k in zprobe_objects:
      zprobe_ids.append(k.client_id)
    # Loads the T-Norm models
    for tmodel_id in tmodel_ids:
      # test if the file is already there
      score_file = self.m_file_selector.d_same_value_file(tmodel_id, group)
      if self.__check_file__(score_file, force):
        print("Score file '%s' already exists." % (score_file))
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
    if tool.m_config.COSINE_SCORING: scoring_type = 'cosine' 
    print("Scoring type = %s" %scoring_type)
    if scoring_type == 'plda' and hasattr(tool, 'load_plda_enroler'):
      # read the model enrolment file
      tool.load_plda_enroler(self.m_file_selector.plda_enroler_file())
    
    # save tool for internal use
    self.m_tool = tool
    self.m_use_projected_ivector_dir = hasattr(tool, 'project_ivector')
    self.m_use_projected_ubm_dir = hasattr(tool, 'project_gmm')
    for group in groups:
      print("----- computing scores for group '%s' -----" % group)
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
        print("computing A scores")
        self.__scores_a__(model_ids_short, group, compute_zt_norm, dir_type, force, preload_probes, scoring_type)
      if compute_zt_norm:
        # compute B scores
        if 'B' in types:
          if indices != None: 
            model_ids_short = model_ids[indices[0]:indices[1]]
          else:
            model_ids_short = model_ids
          print("computing B scores")
          self.__scores_b__(model_ids_short, group, dir_type, force, preload_probes, scoring_type)
        # compute C scores
        if 'C' in types:
          if indices != None: 
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          print("computing C scores")
          self.__scores_c__(tmodel_ids_short, group, dir_type, force, preload_probes, scoring_type)
        # compute D scores
        if 'D' in types:
          if indices != None: 
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          print("computing D scores")
          self.__scores_d__(tmodel_ids_short, group, dir_type, force, preload_probes, scoring_type)
      



  # Function 16/
  def zt_norm(self, tool, groups = ['dev', 'eval']):
    """Computes ZT-Norm using the previously generated files"""
    for group in groups:
      self.m_use_projected_ivector_dir = hasattr(tool, 'project_ivector')
      self.m_use_projected_ubm_dir = hasattr(tool, 'project_gmm')
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
