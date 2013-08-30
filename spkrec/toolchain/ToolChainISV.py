#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:49:54 CEST 2013
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

class ToolChainISV(ToolChain):
  """This class includes functionalities for a default tool chain to produce verification scores"""
   

  def project_isv_features(self, tool, extractor, indices = None, force=False):
    """Extract the features for all files of the database"""
    self.m_tool = tool
    tool_type = self.select_tool_type(tool)

    # load the projector file
    if hasattr(tool, 'project_isv'):
      if hasattr(tool, 'load_projector'):
        tool.load_projector(self.m_file_selector.projector_file())
        
      if hasattr(tool, 'load_enroler'):
        tool.load_enroler(self.m_file_selector.enroler_file())
      
      feature_files = self.m_file_selector.feature_list(tool_type)
      projected_ubm_files = self.m_file_selector.projected_ubm_list(tool_type)
      projected_isv_files = self.m_file_selector.projected_isv_list(tool_type)
      # extract the features
      if indices != None:
        index_range = range(indices[0], indices[1])
        print("- Projection: splitting of index range %s" % str(indices))
      else:
        index_range = range(len(feature_files))

      print("project %d features from directory %s to directory %s using ISV Enroler" %(len(index_range), self.m_file_selector.m_config.projected_ubm_dir, self.m_file_selector.m_config.projected_isv_dir))
      for k in index_range:
        feature_file = feature_files[k]
        projected_ubm_file = projected_ubm_files[k]
        projected_isv_file = projected_isv_files[k]
        
        if not self.__check_file__(projected_isv_file, force):
          # load feature
          feature = self.__read_feature__(feature_file, extractor)
          
          # load projected_ubm file
          projected_ubm = bob.machine.GMMStats(bob.io.HDF5File(str(projected_ubm_file)))
          
          # project isv feature
          projected_isv = tool.project_isv(feature, projected_ubm)
          # write it
          utils.ensure_dir(os.path.dirname(projected_isv_file))
          self.__save_feature__(projected_isv, str(projected_isv_file))
  


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
    use_projected_features = hasattr(tool, 'project_gmm') and not hasattr(tool, 'use_unprojected_features_for_model_enrol')
    # which tool to use to read the features...
    self.m_tool = tool if use_projected_features else extractor

    # Create Models
    if 'N' in types:
      for group in groups:
        model_ids = self.m_file_selector.model_ids(group)

        if indices != None: 
          model_ids = model_ids[indices[0]:indices[1]]
          print("Splitting of index range", indices, "to",)
  
        print("enrol models of group %s" %group)
        for model_id in model_ids:
          # Path to the model
          model_file = self.m_file_selector.model_file(model_id, group)

          # Removes old file if required
          if not self.__check_file__(model_file, force):
            dir_type = 'projected_ubm' if use_projected_features else 'features'
            enrol_files = self.m_file_selector.enrol_files(model_id, group, dir_type)
            
            # load all files into memory
            enrol_features = []
            for k in enrol_files:
              # processes one file
              if os.path.exists(str(k)):
                feature = self.__read_feature__(str(k))
                enrol_features.append(feature)
              else:
                print("Warning: something is wrong with this file: ", str(k))
            
            model = tool.enroll(enrol_features)
            # save the model
            self.__save_model__(model, model_file, tool)

    # T-Norm-Models
    if 'T' in types and compute_zt_norm:
      for group in groups:
        model_ids = self.m_file_selector.tmodel_ids(group)

        if indices != None: 
          model_ids = model_ids[indices[0]:indices[1]]
          print("Splitting of index range", indices, "to",)
  
        print("enrol T-models of group %s" %group)
        for model_id in model_ids:
          # Path to the model
          model_file = self.m_file_selector.tmodel_file(model_id, group)

          # Removes old file if required
          if not self.__check_file__(model_file, force):
            dir_type = 'projected_ubm' if use_projected_features else 'features'
            enrol_files = self.m_file_selector.tenrol_files(model_id, group, dir_type)

            # load all files into memory
            enrol_features = []
            for k in enrol_files:
              # processes one file
              
              feature = self.__read_feature__(str(k))
              enrol_features.append(feature)
              
            model = tool.enroll(enrol_features)
            # save model
            self.__save_model__(model, model_file, tool)



  def __read_model__(self, model_file):
    """This function reads the model from file. Overload this function if your model is no numpy.ndarray."""
    if hasattr(self.m_tool, 'read_model'):
      return self.m_tool.read_model(str(model_file))
    else:
      return bob.io.load(str(model_file))
    

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
    
    


  def __scores_a__(self, model_ids, group, compute_zt_norm, force, preload_probes):
    """Computes A scores"""
    # preload the probe files for a faster access (and fewer network load)
    if self.m_use_projected_isv_dir:
      dir_type = 'projected_isv'
    elif self.m_use_projected_ubm_dir:
      dir_type = 'projected_ubm'
    else:
      dir_type = 'features'
    
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

        if self.m_use_projected_isv_dir:
          dir_type = 'projected_isv'
        elif self.m_use_projected_ubm_dir:
          dir_type = 'projected_ubm'
        else:
          dir_type = 'features'
        probe_files = self.m_file_selector.probe_files_for_model(model_id, group, dir_type)
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


  def __scores_b__(self, model_ids, group, force, preload_probes):
    """Computes B scores"""
    # probe files:
    if self.m_use_projected_isv_dir:
      dir_type = 'projected_isv'
    elif self.m_use_projected_ubm_dir:
      dir_type = 'projected_ubm'
    else:
      dir_type = 'features'
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
        model = self.__read_model__(self.m_file_selector.model_file(model_id, group))
        if preload_probes:
          b = self.__scores_preloaded__(model, zprobes)
        else:
          b = self.__scores__(model, zprobe_objects)
        bob.io.save(b, score_file)

  def __scores_c__(self, tmodel_ids, group, force, preload_probes):
    """Computed C scores"""
    # probe files:
    if self.m_use_projected_isv_dir:
      dir_type = 'projected_isv'
    elif self.m_use_projected_ubm_dir:
      dir_type = 'projected_ubm'
    else:
      dir_type = 'features'
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
        tmodel = self.__read_model__(self.m_file_selector.tmodel_file(tmodel_id, group))
        if preload_probes:
          c = self.__scores_preloaded__(tmodel, probes)
        else:
          c = self.__scores__(tmodel, probe_files)
        bob.io.save(c, score_file)
      
  def __scores_d__(self, tmodel_ids, group, force, preload_probes):
    # probe files:
    if self.m_use_projected_isv_dir:
      dir_type = 'projected_isv'
    elif self.m_use_projected_ubm_dir:
      dir_type = 'projected_ubm'
    else:
      dir_type = 'features'
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
      #zprobe_ids.append(zprobe_objects[k][0]) 
      
    # Loads the T-Norm models
    for tmodel_id in tmodel_ids:
      # test if the file is already there
      score_file = self.m_file_selector.d_same_value_file(tmodel_id, group)
      if self.__check_file__(score_file, force):
        print("Score file '%s' already exists." % (score_file))
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

    self.m_use_projected_ubm_dir = hasattr(tool, 'project_gmm')
    
    # load the projector, if needed
    if hasattr(tool,'load_projector'):
      tool.load_projector(self.m_file_selector.projector_file())
    if hasattr(tool,'load_enroler'):
      tool.load_enroler(self.m_file_selector.enroler_file())

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
        self.__scores_a__(model_ids_short, group, compute_zt_norm, force, preload_probes)
      
      if compute_zt_norm:
        # compute B scores
        if 'B' in types:
          if indices != None: 
            model_ids_short = model_ids[indices[0]:indices[1]]
          else:
            model_ids_short = model_ids
          print("computing B scores")
          self.__scores_b__(model_ids_short, group, force, preload_probes)
        
        # compute C scores
        if 'C' in types:
          if indices != None: 
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          print("computing C scores")
          self.__scores_c__(tmodel_ids_short, group, force, preload_probes)
        
        # compute D scores
        if 'D' in types:
          if indices != None: 
            tmodel_ids_short = tmodel_ids[indices[0]:indices[1]]
          else:
            tmodel_ids_short = tmodel_ids
          print("computing D scores")
          self.__scores_d__(tmodel_ids_short, group, force, preload_probes)
      


  


  def zt_norm(self, tool, groups = ['dev', 'eval']):
    """Computes ZT-Norm using the previously generated files"""
    for group in groups:
      self.m_use_projected_isv_dir = hasattr(tool, 'project_isv')
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
        
