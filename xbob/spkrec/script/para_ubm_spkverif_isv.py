#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Laurent El Shafey <laurent.el-shafey@idiap.ch>
# Manuel Guenther <manuel.guenther@idiap.ch>
# Fri Aug 30 11:45:26 CEST 2013
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

import sys, os
import argparse

from . import ToolChainExecutor
from .. import toolchain
from . import ParallelUBMGMM
import facereclib.utils as utils

class ToolChainExecutorParallelISV (ToolChainExecutor.ToolChainExecutor, ParallelUBMGMM.ParallelUBMGMM):
  """Class that executes the ZT tool chain (locally or in the grid)"""
  
  def __init__(self, args):
    # call base class constructor
    ToolChainExecutor.ToolChainExecutor.__init__(self, args)

    # specify the file selector and tool chain objects to be used by this class (and its base class) 
    self.m_file_selector = toolchain.FileSelector(self.m_configuration, self.m_database_config)
    self.m_tool_chain = toolchain.ToolChainISV(self.m_file_selector)
    ParallelUBMGMM.ParallelUBMGMM.__init__(self)
  
  def parallel_gmm_training_configuration(self):
    """Special configuration specific for parallel UBM-GMM training"""  
    self.m_configuration.normalized_directory = os.path.join(self.m_configuration.base_output_TEMP_dir, 'normalized_features')
    self.m_configuration.kmeans_file = os.path.join(self.m_configuration.base_output_TEMP_dir, 'k_means.hdf5')
    self.m_configuration.kmeans_intermediate_file = os.path.join(self.m_configuration.base_output_TEMP_dir, 'kmeans_temp', 'i_%05d', 'k_means.hdf5')
    self.m_configuration.kmeans_stats_file = os.path.join(self.m_configuration.base_output_TEMP_dir, 'kmeans_temp', 'i_%05d', 'stats_%05d-%05d.hdf5')
    #self.m_tool.m_gmm_filename = os.path.join(self.m_configuration.base_output_TEMP_dir, 'gmm.hdf5')
    self.m_configuration.gmm_intermediate_file = os.path.join(self.m_configuration.base_output_TEMP_dir, 'gmm_temp', 'i_%05d', 'gmm.hdf5')
    self.m_configuration.gmm_stats_file = os.path.join(self.m_configuration.base_output_TEMP_dir, 'gmm_temp', 'i_%05d', 'stats_%05d-%05d.hdf5')
    
  def zt_norm_configuration(self):
    """Special configuration specific for ZT-Norm computation"""
    if self.m_database_config.protocol is not None:
      self.m_configuration.models_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.models_dirs[0], self.m_database_config.protocol)
      self.m_configuration.tnorm_models_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.models_dirs[1], self.m_database_config.protocol)    
      self.m_configuration.zt_norm_A_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[0])
      self.m_configuration.zt_norm_B_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[1])
      self.m_configuration.zt_norm_C_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[2])
      self.m_configuration.zt_norm_D_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[3])
      self.m_configuration.zt_norm_D_sameValue_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.zt_dirs[4])
      self.m_configuration.scores_nonorm_dir = os.path.join(self.m_configuration.base_output_USER_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.score_dirs[0]) 
      self.m_configuration.scores_ztnorm_dir = os.path.join(self.m_configuration.base_output_USER_dir, self.m_args.score_sub_dir, self.m_database_config.protocol, self.m_args.score_dirs[1]) 
    else: 
      self.m_configuration.models_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.models_dirs[0])
      self.m_configuration.tnorm_models_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.models_dirs[1])    
      self.m_configuration.zt_norm_A_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_args.zt_dirs[0])
      self.m_configuration.zt_norm_B_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_args.zt_dirs[1])
      self.m_configuration.zt_norm_C_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_args.zt_dirs[2])
      self.m_configuration.zt_norm_D_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_args.zt_dirs[3])
      self.m_configuration.zt_norm_D_sameValue_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.score_sub_dir, self.m_args.zt_dirs[4])
      self.m_configuration.scores_nonorm_dir = os.path.join(self.m_configuration.base_output_USER_dir, self.m_args.score_sub_dir, self.m_args.score_dirs[0]) 
      self.m_configuration.scores_ztnorm_dir = os.path.join(self.m_configuration.base_output_USER_dir, self.m_args.score_sub_dir, self.m_args.score_dirs[1])  
  
  def isv_specific_configuration(self):
    self.m_configuration.projected_isv_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.projected_isv_dir)
    
  def protocol_specific_configuration(self):
    """Special configuration specific for this toolchain"""
    self.parallel_gmm_training_configuration()
    self.zt_norm_configuration()  
    self.isv_specific_configuration()
    self.m_configuration.default_extension = ".hdf5"
        
  def execute_tool_chain(self):
    """Executes the ZT tool chain on the local machine"""
    # preprocessing
    if not self.m_args.skip_preprocessing:
      self.m_tool_chain.preprocess_audio_files(self.m_preprocessor, self.m_tool, force = self.m_args.force)
    # feature extraction
    #if not self.m_args.skip_feature_extraction_training and hasattr(self.m_feature_extractor, 'train'):
    #  self.m_tool_chain.train_extractor(self.m_feature_extractor, force = self.m_args.force)
    if not self.m_args.skip_feature_extraction:
      self.m_tool_chain.extract_features(self.m_feature_extractor, self.m_tool, force = self.m_args.force)
    
    # training projector
    if not self.m_args.skip_projection_training and hasattr(self.m_tool, 'train_projector'):
      self.m_tool_chain.train_projector(self.m_tool, force = self.m_args.force)
    
    # feature projection    
    if not self.m_args.skip_projection_ubm and hasattr(self.m_tool, 'project_gmm'):
      self.m_tool_chain.project_gmm_features(self.m_tool, force = self.m_args.force, extractor = self.m_feature_extractor)
    
   
    # train enroler
    if not self.m_args.skip_enroler_training and hasattr(self.m_tool, 'train_enroler'):
      self.m_tool_chain.train_enroler(self.m_tool, force = self.m_args.force)
    
    # model enrolment
    if not self.m_args.skip_model_enrolment:
      self.m_tool_chain.enrol_models(self.m_tool, self.m_feature_extractor, not self.m_args.no_zt_norm, groups = self.m_args.groups, force = self.m_args.force)
    
    # ISV projection
    if not self.m_args.skip_projection_isv and hasattr(self.m_tool, 'project_isv'):
      self.m_tool_chain.project_isv_features(self.m_tool, force = self.m_args.force, extractor = self.m_feature_extractor)
    
    # score computation
    if not self.m_args.skip_score_computation:
      self.m_tool_chain.compute_scores(self.m_tool, not self.m_args.no_zt_norm, groups = self.m_args.groups, preload_probes = self.m_args.preload_probes, force = self.m_args.force)
      if not self.m_args.no_zt_norm:
        self.m_tool_chain.zt_norm(self.m_tool, groups = self.m_args.groups)
    
    # concatenation of scores
    if not self.m_args.skip_concatenation:
      self.m_tool_chain.concatenate(not self.m_args.no_zt_norm, groups = self.m_args.groups)
  
    
  def add_jobs_to_grid(self, external_dependencies):
    """Adds all (desired) jobs of the tool chain to the grid"""
    # collect the job ids
    job_ids = {}
  
    # if there are any external dependencies, we need to respect them
    deps = external_dependencies[:]
    
    # VAD; never has any dependencies.
    if not self.m_args.skip_preprocessing:
      job_ids['preprocessing'] = self.submit_grid_job(
              'preprocess', 
              list_to_split = self.m_file_selector.original_wav_list('ISV'), 
              number_of_files_per_job = self.m_grid_config.number_of_audio_files_per_job, 
              dependencies = [], 
              **self.m_grid_config.preprocessing_queue)
      deps.append(job_ids['preprocessing'])
    
    # feature extraction
    if not self.m_args.skip_feature_extraction:
      job_ids['feature-extraction'] = self.submit_grid_job(
              'feature-extraction', 
              list_to_split = self.m_file_selector.feature_list('ISV'), 
              number_of_files_per_job = self.m_grid_config.number_of_audio_files_per_job, 
              dependencies = deps, 
              **self.m_grid_config.extraction_queue)
      deps.append(job_ids['feature-extraction'])      

   
   # feature normalization
    if self.m_args.normalize_features and not self.m_args.skip_normalization:
      job_ids['feature-normalization'] = self.submit_grid_job(
              'normalize-features',
              name="norm-f",
              list_to_split = self.m_file_selector.training_feature_list(),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps,
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['feature-normalization'])


    ######################################################################################
    ############            FINISH THE PARALLEL UBM TRAINING               ###############
    ######################################################################################
    if not self.m_args.skip_projection_training:
      # KMeans
      if not self.m_args.skip_k_means:
        # initialization
        if not self.m_args.kmeans_start_iteration:
          job_ids['kmeans-init'] = self.submit_grid_job(
                  'kmeans-init',
                  name = 'k-init',
                  dependencies = deps,
                  **self.m_grid_config.training_queue)
          deps.append(job_ids['kmeans-init'])

        # several iterations of E and M steps
        for iteration in range(self.m_args.kmeans_start_iteration, self.m_args.kmeans_training_iterations):
          # E-step
          job_ids['kmeans-e-step'] = self.submit_grid_job(
                  'kmeans-e-step --iteration %d' % iteration,
                  name='k-e-%d' % iteration,
                  list_to_split = self.m_file_selector.training_feature_list(),
                  number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
                  dependencies = [job_ids['kmeans-m-step']] if iteration != self.m_args.kmeans_start_iteration else deps,
                  **self.m_grid_config.projection_queue)
  
          # M-step
          job_ids['kmeans-m-step'] = self.submit_grid_job(
                'kmeans-m-step --iteration %d' % iteration,
                name='k-m-%d' % iteration,
                dependencies = [job_ids['kmeans-e-step']],
                **self.m_grid_config.training_queue)

        # add dependence to the last m step
        deps.append(job_ids['kmeans-m-step'])


      # GMM
      if not self.m_args.skip_gmm:
        # initialization
        if not self.m_args.gmm_start_iteration:
          job_ids['gmm-init'] = self.submit_grid_job(
                  'gmm-init',
                  name = 'g-init',
                  dependencies = deps,
                  **self.m_grid_config.training_queue)
          deps.append(job_ids['gmm-init'])
  
        # several iterations of E and M steps
        for iteration in range(self.m_args.gmm_start_iteration, self.m_args.gmm_training_iterations):
          # E-step
          job_ids['gmm-e-step'] = self.submit_grid_job(
                  'gmm-e-step --iteration %d' % iteration,
                  name='g-e-%d' % iteration,
                  list_to_split = self.m_file_selector.training_feature_list(),
                  number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
                  dependencies = [job_ids['gmm-m-step']] if iteration != self.m_args.gmm_start_iteration else deps,
                  **self.m_grid_config.projection_queue)
  
          # M-step
          job_ids['gmm-m-step'] = self.submit_grid_job(
                  'gmm-m-step --iteration %d' % iteration,
                  name='g-m-%d' % iteration,
                  dependencies = [job_ids['gmm-e-step']],
                  **self.m_grid_config.training_queue)
  
        # add dependence to the last m step
        deps.append(job_ids['gmm-m-step'])
           
    ######################################################################################
    ############            FINISH THE PARALLEL UBM TRAINING               ###############
    ######################################################################################

    # feature UBM projection
    if not self.m_args.skip_projection_ubm and hasattr(self.m_tool, 'project_gmm'):
      job_ids['feature_projection_ubm'] = self.submit_grid_job(
              'projection_ubm', 
              list_to_split = self.m_file_selector.feature_list('ISV'),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps, 
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['feature_projection_ubm'])
    
      
    # model enrolment training
    if not self.m_args.skip_enroler_training and hasattr(self.m_tool, 'train_enroler'):
      job_ids['enrolment_training'] = self.submit_grid_job(
              'train_enroler', 
              name = "e-training",
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['enrolment_training'])
       
    # feature ISV projection
    if not self.m_args.skip_projection_isv and hasattr(self.m_tool, 'project_isv'):
      job_ids['feature_projection_isv'] = self.submit_grid_job(
              'projection_isv', 
              list_to_split = self.m_file_selector.feature_list('ISV'),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps, 
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['feature_projection_isv'])
      
    # enrol models
    enrol_deps_n = {}
    enrol_deps_t = {}
    score_deps = {}
    concat_deps = {}
    for group in self.m_args.groups:
      enrol_deps_n[group] = deps[:]
      enrol_deps_t[group] = deps[:]
      list_to_split = self.m_file_selector.model_ids(group)

      if not self.m_args.skip_model_enrolment:
        job_ids['enrol_%s_N'%group] = self.submit_grid_job(
                'enrol_models --group=%s --model-type=N'%group, 
                name = "enrol-N-%s"%group,  
                list_to_split = self.m_file_selector.model_ids(group), 
                number_of_files_per_job = self.m_grid_config.number_of_models_per_enrol_job, 
                dependencies = deps, 
                **self.m_grid_config.enrol_queue)
        enrol_deps_n[group].append(job_ids['enrol_%s_N'%group])
  
        if not self.m_args.no_zt_norm:
          job_ids['enrol_%s_T'%group] = self.submit_grid_job(
                  'enrol_models --group=%s --model-type=T'%group,
                  name = "enrol-T-%s"%group, 
                  list_to_split = self.m_file_selector.tmodel_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_enrol_job, 
                  dependencies = deps,
                  **self.m_grid_config.enrol_queue)
          enrol_deps_t[group].append(job_ids['enrol_%s_T'%group])
          
      # compute A,B,C, and D scores
      if not self.m_args.skip_score_computation:
        job_ids['score_%s_A'%group] = self.submit_grid_job(
                'compute_scores --group=%s --score-type=A'%group, 
                name = "score-A-%s"%group, 
                list_to_split = self.m_file_selector.model_ids(group), 
                number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                dependencies = enrol_deps_n[group], 
                **self.m_grid_config.score_queue)
        concat_deps[group] = [job_ids['score_%s_A'%group]]
        
        if not self.m_args.no_zt_norm:
          job_ids['score_%s_B'%group] = self.submit_grid_job(
                  'compute_scores --group=%s --score-type=B'%group, 
                  name = "score-B-%s"%group, 
                  list_to_split = self.m_file_selector.model_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                  dependencies = enrol_deps_n[group], 
                  **self.m_grid_config.score_queue)
          
          job_ids['score_%s_C'%group] = self.submit_grid_job(
                  'compute_scores --group=%s --score-type=C'%group, 
                  name = "score-C-%s"%group, 
                  list_to_split = self.m_file_selector.tmodel_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                  dependencies = enrol_deps_t[group], 
                  **self.m_grid_config.score_queue)
                  
          job_ids['score_%s_D'%group] = self.submit_grid_job(
                  'compute_scores --group=%s --score-type=D'%group, 
                  name = "score-D-%s"%group, 
                  list_to_split = self.m_file_selector.tmodel_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                  dependencies = enrol_deps_t[group], 
                  **self.m_grid_config.score_queue)
          
          # compute zt-norm
          score_deps[group] = [job_ids['score_%s_A'%group], job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group]]
          job_ids['score_%s_Z'%group] = self.submit_grid_job(
                  'compute_scores --group=%s --score-type=Z'%group,
                  name = "score-Z-%s"%group,
                  dependencies = score_deps[group], **self.m_grid_config.score_queue) 
          concat_deps[group].extend([job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group], job_ids['score_%s_Z'%group]])
      else:
        concat_deps[group] = []

      # concatenate results   
      if not self.m_args.skip_concatenation:
        job_ids['concat_%s'%group] = self.submit_grid_job(
                'concatenate --group=%s'%group,
                name = "concat-%s"%group,
                dependencies = concat_deps[group])
        
    # return the job ids, in case anyone wants to know them
    return job_ids 
  

  def execute_grid_job(self):
    """Run the desired job of the ZT tool chain that is specified on command line""" 
    # preprocess
    if self.m_args.execute_sub_task == 'preprocess':
      self.m_tool_chain.preprocess_audio_files(
          self.m_preprocessor, 
          self.m_tool,
          indices = self.indices(self.m_file_selector.original_wav_list('ISV'), self.m_grid_config.number_of_audio_files_per_job), 
          force = self.m_args.force)
    
    # feature extraction
    if self.m_args.execute_sub_task == 'feature-extraction':
      self.m_tool_chain.extract_features(
          self.m_feature_extractor, 
          self.m_tool,
          indices = self.indices(self.m_file_selector.feature_list('ISV'), self.m_grid_config.number_of_audio_files_per_job), 
          force = self.m_args.force)
      
   
    # normalize features
    elif self.m_args.execute_sub_task == 'normalize-features':
      self.feature_normalization(
          indices = self.indices(self.m_file_selector.training_feature_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # kmeans init
    elif self.m_args.execute_sub_task == 'kmeans-init':
      self.kmeans_initialize(
          force = self.m_args.force)

    # kmeans E-step
    elif self.m_args.execute_sub_task == 'kmeans-e-step':
      self.kmeans_estep(
          indices = self.indices(self.m_file_selector.training_feature_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # Kmeans M-step
    elif self.m_args.execute_sub_task == 'kmeans-m-step':
      self.kmeans_mstep(
          counts = self.m_grid_config.number_of_projections_per_job,
          force = self.m_args.force)

    # ML Learning init
    elif self.m_args.execute_sub_task == 'gmm-init':
      self.gmm_initialize(
          force = self.m_args.force)

    # ML Learning E-step
    elif self.m_args.execute_sub_task == 'gmm-e-step':
      self.gmm_estep(
          indices = self.indices(self.m_file_selector.training_feature_list(), self.m_grid_config.number_of_projections_per_job),
          force = self.m_args.force)

    # ML Learning M-step
    elif self.m_args.execute_sub_task == 'gmm-m-step':
      self.gmm_mstep(
          counts = self.m_grid_config.number_of_projections_per_job,
          force = self.m_args.force)

      
    # project the features ubm
    elif self.m_args.execute_sub_task == 'projection_ubm':
      self.m_tool_chain.project_gmm_features(
          self.m_tool, 
          indices = self.indices(self.m_file_selector.feature_list('ISV'), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force,
          extractor = self.m_feature_extractor)
     
    # train model enroler
    elif self.m_args.execute_sub_task == 'train_enroler':
      self.m_tool_chain.train_enroler(
          self.m_tool, 
          force = self.m_args.force)
    
    # project the features isv
    elif self.m_args.execute_sub_task == 'projection_isv':
      self.m_tool_chain.project_isv_features(
          self.m_tool, 
          indices = self.indices(self.m_file_selector.feature_list('ISV'), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force,
          extractor = self.m_feature_extractor)
      
    # enrol models
    elif self.m_args.execute_sub_task == 'enrol_models':
      if self.m_args.model_type == 'N':
        self.m_tool_chain.enrol_models(
            self.m_tool,
            self.m_feature_extractor,
            not self.m_args.no_zt_norm, 
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid_config.number_of_models_per_enrol_job), 
            groups = [self.m_args.group], 
            types = ['N'], 
            force = self.m_args.force)

      else:
        self.m_tool_chain.enrol_models(
            self.m_tool,
            self.m_feature_extractor,
            not self.m_args.no_zt_norm, 
            indices = self.indices(self.m_file_selector.tmodel_ids(self.m_args.group), self.m_grid_config.number_of_models_per_enrol_job), 
            groups = [self.m_args.group], 
            types = ['T'], 
            force = self.m_args.force)
        
    # compute scores
    elif self.m_args.execute_sub_task == 'compute_scores':
      if self.m_args.score_type in ['A', 'B']:
        self.m_tool_chain.compute_scores(
            self.m_tool, 
            not self.m_args.no_zt_norm, 
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid_config.number_of_models_per_score_job), 
            groups = [self.m_args.group], 
            types = [self.m_args.score_type], 
            preload_probes = self.m_args.preload_probes, 
            force = self.m_args.force)

      elif self.m_args.score_type in ['C', 'D']:
        self.m_tool_chain.compute_scores(
            self.m_tool, 
            not self.m_args.no_zt_norm, 
            indices = self.indices(self.m_file_selector.tmodel_ids(self.m_args.group), self.m_grid_config.number_of_models_per_score_job), 
            groups = [self.m_args.group], 
            types = [self.m_args.score_type], 
            preload_probes = self.m_args.preload_probes, 
            force = self.m_args.force)

      else:
        self.m_tool_chain.zt_norm(self.m_tool, groups = [self.m_args.group])
    # concatenate
    if self.m_args.execute_sub_task == 'concatenate':
      self.m_tool_chain.concatenate(
          not self.m_args.no_zt_norm, 
          groups = [self.m_args.group])



def parse_args(command_line_arguments = sys.argv[1:]):
  """This function parses the given options (which by default are the command line options)"""
  # sorry for that.
  global parameters
  parameters = command_line_arguments

  # set up command line parser
  parser = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # add the arguments required for all tool chains
  config_group, dir_group, file_group, sub_dir_group, other_group, skip_group = ToolChainExecutorParallelISV.required_command_line_options(parser)

  skip_group.add_argument('--skip-projection-isv', '--noproisv', action='store_true', dest='skip_projection_isv',
        help = 'If using ISV Tool, skip the feature ISV projection')  

  sub_dir_group.add_argument('--projected-isv-directory', type = str, metavar = 'DIR', default = 'projected_isv', dest = 'projected_isv_dir',
      help = 'Name of the directory where the projected data should be stored')
  sub_dir_group.add_argument('--models-directories', type = str, metavar = 'DIR', nargs = 2, dest='models_dirs',
      default = ['models', 'tmodels'],
      help = 'Subdirectories (of temp directory) where the models should be stored')
  sub_dir_group.add_argument('--zt-norm-directories', type = str, metavar = 'DIR', nargs = 5, dest='zt_dirs', 
      default = ['zt_norm_A', 'zt_norm_B', 'zt_norm_C', 'zt_norm_D', 'zt_norm_D_sameValue'],
      help = 'Subdirectories (of --temp-dir) where to write the zt_norm values')
  sub_dir_group.add_argument('--score-dirs', type = str, metavar = 'DIR', nargs = 2, dest='score_dirs',
      default = ['nonorm', 'ztnorm'],
      help = 'Subdirectories (of --user-dir) where to write the results to')
    
  #######################################################################################
  ############################ other options ############################################
  other_group.add_argument('-z', '--no-zt-norm', action='store_true', dest = 'no_zt_norm',
      help = 'DISABLE the computation of ZT norms')
  other_group.add_argument('-F', '--force', action='store_true',
      help = 'Force to erase former data if already exist')
  other_group.add_argument('-w', '--preload-probes', action='store_true', dest='preload_probes',
      help = 'Preload probe files during score computation (needs more memory, but is faster and requires fewer file accesses). WARNING! Use this flag with care!')
  other_group.add_argument('--groups', type = str,  metavar = 'GROUP', nargs = '+', default = ['dev', 'eval'],
      help = "The group (i.e., 'dev' or  'eval') for which the models and scores should be generated")
  other_group.add_argument('-l', '--limit-training-examples', type=int,
      help = 'Limit the number of training examples used for KMeans initialization and the GMM initialization')

  other_group.add_argument('-K', '--kmeans-training-iterations', type=int, default=25,
      help = 'Specify the number of training iterations for the KMeans training')
  other_group.add_argument('-k', '--kmeans-start-iteration', type=int, default=0,
      help = 'Specify the first iteration for the KMeans training (i.e. to restart)')

  other_group.add_argument('-M', '--gmm-training-iterations', type=int, default=25,
      help = 'Specify the number of training iterations for the GMM training')
  other_group.add_argument('-m', '--gmm-start-iteration', type=int, default=0,
      help = 'Specify the first iteration for the GMM training (i.e. to restart)')
  other_group.add_argument('-n', '--normalize-features', action='store_true',
      help = 'Normalize features before UBM-GMM training?')
  other_group.add_argument('-C', '--clean-intermediate', action='store_true',
      help = 'Clean up temporary files of older iterations?')

  skip_group.add_argument('--skip-normalization', '--non', action='store_true',
      help = "Skip the feature normalization step")
  skip_group.add_argument('--skip-k-means', '--nok', action='store_true',
      help = "Skip the KMeans step")
  skip_group.add_argument('--skip-gmm', '--nog', action='store_true',
      help = "Skip the GMM step")
#  skip_group.add_argument('--skip-gmm-projection', '--nogp', action='store_true',
#      help = "Skip the GMM projection step")
#  skip_group.add_argument('--skip-isv', '--noi', action='store_true',
#      help = "Skip the ISV step")
#  skip_group.add_argument('--skip-isv-projection', '--noip', action='store_true',
#      help = "Skip the GMM isv projection")



  #######################################################################################
  #################### sub-tasks being executed by this script ##########################

  parser.add_argument('--execute-sub-task',
      choices = ('preprocess', 'feature-extraction', 'normalize-features', 'kmeans-init', 'kmeans-e-step', 'kmeans-m-step', 'gmm-init', 'gmm-e-step', 'gmm-m-step', 'projection_ubm', 'train_enroler', 'projection_isv', 'enrol_models', 'compute_scores',  'concatenate'),
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--iteration', type=int,
      help = argparse.SUPPRESS) #'The current iteration of KMeans or GMM training' 
      
  parser.add_argument('--model-type', type = str, choices = ['N', 'T'], metavar = 'TYPE', 
      help = argparse.SUPPRESS) #'Which type of models to generate (Normal or TModels)'
  parser.add_argument('--score-type', type = str, choices=['A', 'B', 'C', 'D', 'Z'],  metavar = 'SCORE', 
      help = argparse.SUPPRESS) #'The type of scores that should be computed'
  parser.add_argument('--group', type = str,  metavar = 'GROUP', 
      help = argparse.SUPPRESS) #'The group for which the current action should be performed'

  
  return parser.parse_args(command_line_arguments)


def speaker_verify(args, external_dependencies = [], external_fake_job_id = 0):
  """This is the main entry point for computing speaker verification experiments.
  You just have to specify configuration scripts for any of the steps of the toolchain, which are:
  -- the database
  -- preprocessing (VAD)
  -- feature extraction
  -- the score computation tool
  -- and the grid configuration (in case, the function should be executed in the grid).
  Additionally, you can skip parts of the toolchain by selecting proper --skip-... parameters.
  If your probe files are not too big, you can also specify the --preload-probes switch to speed up the score computation.
  If files should be re-generated, please specify the --force option (might be combined with the --skip-... options)"""
  
  
  # generate tool chain executor
  executor = ToolChainExecutorParallelISV(args)
  # as the main entry point, check whether the grid option was given
  if not args.grid:
    # not in a grid, use default tool chain sequentially
    executor.execute_tool_chain()
    return []
    
  elif args.execute_sub_task:
    # execute the desired sub-task
    executor.execute_grid_job()
    return []
  else:
    # no other parameter given, so deploy new jobs

    # get the name of this file 
    this_file = __file__
    if this_file[-1] == 'c':
      this_file = this_file[0:-1]
      
    # initialize the executor to submit the jobs to the grid 
    global parameters
    executor.set_common_parameters(calling_file = this_file, parameters = parameters, fake_job_id = external_fake_job_id )
    
    # add the jobs
    return executor.add_jobs_to_grid(external_dependencies)
    

def main(command_line_parameters = sys.argv):
  """Executes the main function"""
  # do the command line parsing
  args = parse_args(command_line_parameters[1:])
  # verify that the input files exist
  for f in (args.database, args.preprocessor, args.tool):
    if not os.path.exists(str(f)):
      raise ValueError("The given file '%s' does not exist."%f)
  # perform speaker verification test
  speaker_verify(args)
        
if __name__ == "__main__":
  main()  

