#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <elie.khoury@idiap.ch>
# Fri Aug 30 11:47:56 CEST 2013
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

class ToolChainExecutorIVector (ToolChainExecutor.ToolChainExecutor):
  """Class that executes the I-Vector tool chain (locally or in the grid)"""
  
  def __init__(self, args):
    # call base class constructor
    ToolChainExecutor.ToolChainExecutor.__init__(self, args)

    # specify the file selector and tool chain objects to be used by this class (and its base class) 
    self.m_file_selector = toolchain.FileSelector(self.m_configuration, self.m_database_config)
    self.m_tool_chain = toolchain.ToolChainIvector(self.m_file_selector)
    
    
  def zt_norm_configuration(self):
    """Special configuration for ZT-Norm computation"""
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
    
    self.m_configuration.default_extension = ".hdf5"
  
  
  def ivector_specific_configuration(self):
    self.m_configuration.whitening_enroler_file = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.whitening_enroler_file)
    self.m_configuration.lda_projector_file = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.lda_projector_file)
    self.m_configuration.wccn_projector_file = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.wccn_projector_file)
    self.m_configuration.plda_enroler_file = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.plda_enroler_file)
    self.m_configuration.projected_ivector_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.projected_ivector_dir)
    self.m_configuration.whitened_ivector_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.whitened_ivector_dir)
    self.m_configuration.lnorm_ivector_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.lnorm_ivector_dir)
    self.m_configuration.lda_projected_ivector_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.lda_projected_ivector_dir)
    self.m_configuration.wccn_projected_ivector_dir = os.path.join(self.m_configuration.base_output_TEMP_dir, self.m_args.wccn_projected_ivector_dir)
        
  def protocol_specific_configuration(self):
    """Special configuration specific for this toolchain"""
    self.zt_norm_configuration()  
    self.ivector_specific_configuration()    
    
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
    # feature projection
    if not self.m_args.skip_projection_training and hasattr(self.m_tool, 'train_projector'):
      self.m_tool_chain.train_projector(self.m_tool, force = self.m_args.force)
    
    if not self.m_args.skip_projection_ubm and hasattr(self.m_tool, 'project_gmm'):
      self.m_tool_chain.project_gmm_features(self.m_tool, force = self.m_args.force, extractor = self.m_feature_extractor)
       
    # train enroler
    if not self.m_args.skip_enroler_training and hasattr(self.m_tool, 'train_enroler'):
      self.m_tool_chain.train_enroler(self.m_tool, force = self.m_args.force)

    # IVector projection
    if not self.m_args.skip_projection_ivector and hasattr(self.m_tool, 'project_ivector'):
      self.m_tool_chain.project_ivector_features(self.m_tool, force = self.m_args.force, extractor = self.m_feature_extractor)
    
    # train whitening enroler
    if not self.m_args.skip_whitening_enroler_training and hasattr(self.m_tool, 'train_whitening_enroler'):
      self.m_tool_chain.train_whitening_enroler(self.m_tool, dir_type='projected_ivector', force = self.m_args.force)
    
    # whitening i-vectors
    if not self.m_args.skip_whitening_ivector and hasattr(self.m_tool, 'whitening_ivector'):
      self.m_tool_chain.whitening_ivector(self.m_tool, dir_type='projected_ivector', force = self.m_args.force)
    
    # lnorm i-vectors
    if not self.m_args.skip_lnorm_ivector and hasattr(self.m_tool, 'lnorm_ivector'):
      self.m_tool_chain.lnorm_ivector(self.m_tool, dir_type='whitened_ivector', force = self.m_args.force)
    
    # train LDA projector
    if not self.m_args.skip_lda_train_projector and hasattr(self.m_tool, 'lda_train_projector'):
      self.m_tool_chain.lda_train_projector(self.m_tool, dir_type='lnorm_ivector', force = self.m_args.force)
      
    # project i-vectors using LDA
    if not self.m_args.skip_lda_projection and hasattr(self.m_tool, 'lda_project_ivector'):
      self.m_tool_chain.lda_project_ivector(self.m_tool, dir_type='lnorm_ivector', force = self.m_args.force)  
          
    # train WCCN projector
    if not self.m_args.skip_wccn_train_projector and hasattr(self.m_tool, 'wccn_train_projector'):
      self.m_tool_chain.wccn_train_projector(self.m_tool, dir_type='lda_projected_ivector', force = self.m_args.force)
      
    # project i-vectors using WCCN
    if not self.m_args.skip_wccn_projection and hasattr(self.m_tool, 'wccn_project_ivector'):
      self.m_tool_chain.wccn_project_ivector(self.m_tool, dir_type='lda_projected_ivector', force = self.m_args.force)  
    
    cur_type = 'wccn_projected_ivector'
          
    # train plda enroler
    if not self.m_args.skip_train_plda_enroler and hasattr(self.m_tool, 'train_plda_enroler'):
      self.m_tool_chain.train_plda_enroler(self.m_tool, dir_type=cur_type, force = self.m_args.force)
    
    # PLDA enrollment of the models
    if not self.m_args.skip_model_enrolment:
      self.m_tool_chain.enrol_models(self.m_tool, self.m_feature_extractor, not self.m_args.no_zt_norm, dir_type=cur_type, groups = self.m_args.groups, force = self.m_args.force)
    
      
    # score computation
    if not self.m_args.skip_score_computation:
      self.m_tool_chain.compute_scores(self.m_tool, not self.m_args.no_zt_norm, dir_type=cur_type, groups = self.m_args.groups, preload_probes = self.m_args.preload_probes, force = self.m_args.force)
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
    
    # VAD preprocessing; never has any dependencies.
    if not self.m_args.skip_preprocessing:
      job_ids['preprocessing'] = self.submit_grid_job(
              '--preprocess', 
              list_to_split = self.m_file_selector.original_wav_list('IVector'), 
              number_of_files_per_job = self.m_grid_config.number_of_audio_files_per_job, 
              dependencies = [], 
              **self.m_grid_config.preprocessing_queue)
      deps.append(job_ids['preprocessing'])
    
    # feature extraction
    if not self.m_args.skip_feature_extraction:
      job_ids['feature_extraction'] = self.submit_grid_job(
              '--feature-extraction', 
              list_to_split = self.m_file_selector.feature_list('IVector'), 
              number_of_files_per_job = self.m_grid_config.number_of_audio_files_per_job, 
              dependencies = deps, 
              **self.m_grid_config.extraction_queue)
      deps.append(job_ids['feature_extraction'])      

    # feature projection training
    if not self.m_args.skip_projection_training and hasattr(self.m_tool, 'train_projector'):
      job_ids['projector_training'] = self.submit_grid_job(
              '--train-projector', 
              name="p-training", 
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['projector_training'])
    
    # feature UBM projection
    if not self.m_args.skip_projection_ubm and hasattr(self.m_tool, 'project_gmm'):
      job_ids['feature_projection_ubm'] = self.submit_grid_job(
              '--feature-projection-ubm', 
              list_to_split = self.m_file_selector.feature_list('IVector'),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps, 
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['feature_projection_ubm'])
    
      
    # TV training
    if not self.m_args.skip_enroler_training and hasattr(self.m_tool, 'train_enroler'):
      job_ids['enrolment_training'] = self.submit_grid_job(
              '--train-enroler', 
              name = "e-training",
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['enrolment_training'])
       
    # i-vectors extraction
    if not self.m_args.skip_projection_ivector and hasattr(self.m_tool, 'project_ivector'):
      job_ids['feature_projection_ivector'] = self.submit_grid_job(
              '--feature-projection-ivector', 
              list_to_split = self.m_file_selector.feature_list('IVector'),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps, 
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['feature_projection_ivector'])
    
    
    # train whitening
    if not self.m_args.skip_whitening_enroler_training and hasattr(self.m_tool, 'train_whitening_enroler'):
      job_ids['whitening_enrolment_training'] = self.submit_grid_job(
              '--train-whitening-enroler', 
              name = "w-e-training",
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['whitening_enrolment_training'])
    
    # whitening i-vectors
    if not self.m_args.skip_whitening_ivector and hasattr(self.m_tool, 'whitening_ivector'):
      job_ids['whitening_ivector'] = self.submit_grid_job(
              '--whitening-ivector', 
              list_to_split = self.m_file_selector.feature_list('IVector'),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps, 
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['whitening_ivector'])

    # lnorm i-vectors
    if not self.m_args.skip_lnorm_ivector and hasattr(self.m_tool, 'lnorm_ivector'):
      job_ids['lnorm_ivector'] = self.submit_grid_job(
              '--lnorm-ivector', 
              list_to_split = self.m_file_selector.feature_list('IVector'),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps, 
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['lnorm_ivector'])

    # train LDA projector
    if not self.m_args.skip_lda_train_projector and hasattr(self.m_tool, 'lda_train_projector'):
      job_ids['lda_train_projector'] = self.submit_grid_job(
              '--lda-train-projector', 
              name = "lda-proj-training",
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['lda_train_projector'])
    
    # LDA projection
    if not self.m_args.skip_lda_projection and hasattr(self.m_tool, 'lda_project_ivector'):
      job_ids['lda_project_ivector'] = self.submit_grid_job(
              '--lda-project-ivector', 
              list_to_split = self.m_file_selector.feature_list('IVector'),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps, 
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['lda_project_ivector'])

    # train WCCN projector
    if not self.m_args.skip_wccn_train_projector and hasattr(self.m_tool, 'wccn_train_projector'):
      job_ids['wccn_train_projector'] = self.submit_grid_job(
              '--wccn-train-projector', 
              name = "wccn-proj-training",
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['wccn_train_projector'])
    
    # WCCN projection
    if not self.m_args.skip_wccn_projection and hasattr(self.m_tool, 'wccn_project_ivector'):
      job_ids['wccn_project_ivector'] = self.submit_grid_job(
              '--wccn-project-ivector', 
              list_to_split = self.m_file_selector.feature_list('IVector'),
              number_of_files_per_job = self.m_grid_config.number_of_projections_per_job,
              dependencies = deps, 
              **self.m_grid_config.projection_queue)
      deps.append(job_ids['wccn_project_ivector'])
    
        
    # train PLDA
    if not self.m_args.skip_train_plda_enroler and hasattr(self.m_tool, 'train_plda_enroler'):
      job_ids['train_plda_enroler'] = self.submit_grid_job(
              '--train-plda-enroler', 
              name = "plda-e-training",
              dependencies = deps, 
              **self.m_grid_config.training_queue)
      deps.append(job_ids['train_plda_enroler'])
          
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
                '--enrol-models --group=%s --model-type=N'%group, 
                name = "enrol-N-%s"%group,  
                list_to_split = self.m_file_selector.model_ids(group), 
                number_of_files_per_job = self.m_grid_config.number_of_models_per_enrol_job, 
                dependencies = deps, 
                **self.m_grid_config.enrol_queue)
        enrol_deps_n[group].append(job_ids['enrol_%s_N'%group])
  
        if not self.m_args.no_zt_norm:
          job_ids['enrol_%s_T'%group] = self.submit_grid_job(
                  '--enrol-models --group=%s --model-type=T'%group,
                  name = "enrol-T-%s"%group, 
                  list_to_split = self.m_file_selector.tmodel_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_enrol_job, 
                  dependencies = deps,
                  **self.m_grid_config.enrol_queue)
          enrol_deps_t[group].append(job_ids['enrol_%s_T'%group])
          
      # compute A,B,C, and D scores
      if not self.m_args.skip_score_computation:
        job_ids['score_%s_A'%group] = self.submit_grid_job(
                '--compute-scores --group=%s --score-type=A'%group, 
                name = "score-A-%s"%group, 
                list_to_split = self.m_file_selector.model_ids(group), 
                number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                dependencies = enrol_deps_n[group], 
                **self.m_grid_config.score_queue)
        concat_deps[group] = [job_ids['score_%s_A'%group]]
        
        if not self.m_args.no_zt_norm:
          job_ids['score_%s_B'%group] = self.submit_grid_job(
                  '--compute-scores --group=%s --score-type=B'%group, 
                  name = "score-B-%s"%group, 
                  list_to_split = self.m_file_selector.model_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                  dependencies = enrol_deps_n[group], 
                  **self.m_grid_config.score_queue)
          
          job_ids['score_%s_C'%group] = self.submit_grid_job(
                  '--compute-scores --group=%s --score-type=C'%group, 
                  name = "score-C-%s"%group, 
                  list_to_split = self.m_file_selector.tmodel_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                  dependencies = enrol_deps_t[group], 
                  **self.m_grid_config.score_queue)
                  
          job_ids['score_%s_D'%group] = self.submit_grid_job(
                  '--compute-scores --group=%s --score-type=D'%group, 
                  name = "score-D-%s"%group, 
                  list_to_split = self.m_file_selector.tmodel_ids(group), 
                  number_of_files_per_job = self.m_grid_config.number_of_models_per_score_job, 
                  dependencies = enrol_deps_t[group], 
                  **self.m_grid_config.score_queue)
          
          # compute zt-norm
          score_deps[group] = [job_ids['score_%s_A'%group], job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group]]
          job_ids['score_%s_Z'%group] = self.submit_grid_job(
                  '--compute-scores --group=%s --score-type=Z'%group,
                  name = "score-Z-%s"%group,
                  dependencies = score_deps[group], **self.m_grid_config.score_queue) 
          concat_deps[group].extend([job_ids['score_%s_B'%group], job_ids['score_%s_C'%group], job_ids['score_%s_D'%group], job_ids['score_%s_Z'%group]])
      else:
        concat_deps[group] = []

      # concatenate results   
      if not self.m_args.skip_concatenation:
        job_ids['concat_%s'%group] = self.submit_grid_job(
                '--concatenate --group=%s'%group,
                name = "concat-%s"%group,
                dependencies = concat_deps[group])
        
    # return the job ids, in case anyone wants to know them
    return job_ids 
  

  def execute_grid_job(self):
    """Run the desired job of the ZT tool chain that is specified on command line""" 
    # preprocess
    if self.m_args.preprocess:
      self.m_tool_chain.preprocess_audio_files(
          self.m_preprocessor, 
          self.m_tool,
          indices = self.indices(self.m_file_selector.original_wav_list('IVector'), self.m_grid_config.number_of_audio_files_per_job), 
          force = self.m_args.force)
    
    # feature extraction
    if self.m_args.feature_extraction:
      self.m_tool_chain.extract_features(
          self.m_feature_extractor, 
          self.m_tool,
          indices = self.indices(self.m_file_selector.feature_list('IVector'), self.m_grid_config.number_of_audio_files_per_job), 
          force = self.m_args.force)
      
    # train the feature projector
    if self.m_args.train_projector:
      self.m_tool_chain.train_projector(
          self.m_tool, 
          force = self.m_args.force)
      
    # project the features ubm
    if self.m_args.projection_ubm:
      self.m_tool_chain.project_gmm_features(
          self.m_tool, 
          indices = self.indices(self.m_file_selector.feature_list('IVector'), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force,
          extractor = self.m_feature_extractor)
     
    # train model enroler
    if self.m_args.train_enroler:
      self.m_tool_chain.train_enroler(
          self.m_tool, 
          force = self.m_args.force)
    
    # project the features ivector
    if self.m_args.projection_ivector:
      self.m_tool_chain.project_ivector_features(
          self.m_tool, 
          indices = self.indices(self.m_file_selector.feature_list('IVector'), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force,
          extractor = self.m_feature_extractor)
    
    # train model whitening enroler
    if self.m_args.train_whitening_enroler:
      self.m_tool_chain.train_whitening_enroler(
          self.m_tool, 
          dir_type='projected_ivector',
          force = self.m_args.force)
    
    # project the features ivector
    if self.m_args.whitening_ivector:
      self.m_tool_chain.whitening_ivector(
          self.m_tool, 
          dir_type='projected_ivector',
          indices = self.indices(self.m_file_selector.feature_list('IVector'), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force)

    # project the features ivector
    if self.m_args.lnorm_ivector:
      self.m_tool_chain.lnorm_ivector(
          self.m_tool, 
          dir_type='whitened_ivector',
          indices = self.indices(self.m_file_selector.feature_list('IVector'), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force)
              
    # train LDA projector
    if self.m_args.lda_train_projector:
      self.m_tool_chain.lda_train_projector(
          self.m_tool, 
          dir_type='lnorm_ivector',
          force = self.m_args.force)
    
    # project the features ivector
    if self.m_args.lda_project_ivector:
      self.m_tool_chain.lda_project_ivector(
          self.m_tool, 
          dir_type='lnorm_ivector',
          indices = self.indices(self.m_file_selector.feature_list('IVector'), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force)
      
    
    # train WCCN projector
    if self.m_args.wccn_train_projector:
      self.m_tool_chain.wccn_train_projector(
          self.m_tool, 
          dir_type='lda_projected_ivector',
          force = self.m_args.force)
    
    # project the features ivector
    if self.m_args.wccn_project_ivector:
      self.m_tool_chain.wccn_project_ivector(
          self.m_tool, 
          dir_type='lda_projected_ivector',
          indices = self.indices(self.m_file_selector.feature_list('IVector'), self.m_grid_config.number_of_projections_per_job), 
          force = self.m_args.force)
    
    
    cur_type = 'wccn_projected_ivector'
    
    # train plda enroler
    if self.m_args.train_plda_enroler:
      self.m_tool_chain.train_plda_enroler(
          self.m_tool,
          dir_type=cur_type,
          force = self.m_args.force)
    
      
    # enrol models
    if self.m_args.enrol_models:
      if self.m_args.model_type == 'N':
        self.m_tool_chain.enrol_models(
            self.m_tool,
            self.m_feature_extractor,
            not self.m_args.no_zt_norm, 
            dir_type = cur_type,
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid_config.number_of_models_per_enrol_job), 
            groups = [self.m_args.group], 
            types = ['N'], 
            force = self.m_args.force)

      else:
        self.m_tool_chain.enrol_models(
            self.m_tool,
            self.m_feature_extractor,
            not self.m_args.no_zt_norm, 
            dir_type = cur_type,
            indices = self.indices(self.m_file_selector.tmodel_ids(self.m_args.group), self.m_grid_config.number_of_models_per_enrol_job), 
            groups = [self.m_args.group], 
            types = ['T'], 
            force = self.m_args.force)        
    # compute scores
    if self.m_args.compute_scores:
      if self.m_args.score_type in ['A', 'B']:
        self.m_tool_chain.compute_scores(
            self.m_tool, 
            not self.m_args.no_zt_norm, 
            dir_type = cur_type,
            indices = self.indices(self.m_file_selector.model_ids(self.m_args.group), self.m_grid_config.number_of_models_per_score_job), 
            groups = [self.m_args.group], 
            types = [self.m_args.score_type], 
            preload_probes = self.m_args.preload_probes, 
            force = self.m_args.force)

      elif self.m_args.score_type in ['C', 'D']:
        self.m_tool_chain.compute_scores(
            self.m_tool, 
            not self.m_args.no_zt_norm, 
            dir_type = cur_type,
            indices = self.indices(self.m_file_selector.tmodel_ids(self.m_args.group), self.m_grid_config.number_of_models_per_score_job), 
            groups = [self.m_args.group], 
            types = [self.m_args.score_type], 
            preload_probes = self.m_args.preload_probes, 
            force = self.m_args.force)

      else:
        self.m_tool_chain.zt_norm(self.m_tool, groups = [self.m_args.group])
    # concatenate
    if self.m_args.concatenate:
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
  config_group, dir_group, file_group, sub_dir_group, other_group, skip_group = ToolChainExecutorIVector.required_command_line_options(parser)

  file_group.add_argument('--whitening-enroler-file' , type = str, metavar = 'FILE', default = 'WhiteEnroler.hdf5',
      help = 'Name of the file to write the model of whitening enroler into')
  file_group.add_argument('--lda-projector-file' , type = str, metavar = 'FILE', default = 'LDAProjector.hdf5',
      help = 'Name of the file to write the model of LDA projector into')
  file_group.add_argument('--wccn-projector-file' , type = str, metavar = 'FILE', default = 'WCCNProjector.hdf5',
      help = 'Name of the file to write the model of WCCN projector into')
  file_group.add_argument('--plda-enroler-file' , type = str, metavar = 'FILE', default = 'PLDAEnroler.hdf5',
      help = 'Name of the file to write the model of PLDA enroler into')
  
  sub_dir_group.add_argument('--projected-ivector-directory', type = str, metavar = 'DIR', default = 'projected_ivector', dest = 'projected_ivector_dir',
      help = 'Name of the directory where the projected data should be stored')
  sub_dir_group.add_argument('--whitened-ivector-directory', type = str, metavar = 'DIR', default = 'whitened_ivector', dest = 'whitened_ivector_dir',
      help = 'Name of the directory where the projected data should be stored')
  sub_dir_group.add_argument('--lnorm-ivector-directory', type = str, metavar = 'DIR', default = 'lnorm_ivector', dest = 'lnorm_ivector_dir',
      help = 'Name of the directory where the projected data should be stored')
  sub_dir_group.add_argument('--lda-projected-ivector-directory', type = str, metavar = 'DIR', default = 'lda_projected_ivector', dest = 'lda_projected_ivector_dir',
      help = 'Name of the directory where the projected data should be stored')
  sub_dir_group.add_argument('--wccn-projected-ivector-directory', type = str, metavar = 'DIR', default = 'wccn_projected_ivector', dest = 'wccn_projected_ivector_dir',
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
  
  skip_group.add_argument('--skip-projection-ivector', '--noproivec', action='store_true', dest='skip_projection_ivector',
      help = 'Skip the feature IVector projection')
  skip_group.add_argument('--skip-whitening-enroler-training', '--nowenrt', action='store_true', dest='skip_whitening_enroler_training',
      help = 'Skip the training of the model whitening enrolment')
  skip_group.add_argument('--skip-whitening-ivector', '--nowivec', action='store_true', dest='skip_whitening_ivector',
      help = 'Skip whitening i-vectors')
  skip_group.add_argument('--skip-lnorm-ivector', '--nolnivec', action='store_true', dest='skip_lnorm_ivector',
      help = 'Skip lnorm i-vectors')
  skip_group.add_argument('--skip-lda-train-projector', '--noldaprojt', action='store_true', dest='skip_lda_train_projector',
      help = 'Skip the training of the LDA projector')
  skip_group.add_argument('--skip-lda-projection', '--noldaproj', action='store_true', dest='skip_lda_projection',
      help = 'Skip projecting i-vectors on LDA')
  skip_group.add_argument('--skip-wccn-train-projector', '--nowccnprojt', action='store_true', dest='skip_wccn_train_projector',
      help = 'Skip the training of the WCCN projector')
  skip_group.add_argument('--skip-wccn-projection', '--nowccnproj', action='store_true', dest='skip_wccn_projection',
      help = 'Skip projecting i-vectors on WCCN')
  skip_group.add_argument('--skip-train-plda-enroler', '--nopldaenrt', action='store_true', dest='skip_train_plda_enroler',
      help = 'Skip the training of the plda model enrolment')

  
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

  #######################################################################################
  #################### sub-tasks being executed by this script ##########################
  parser.add_argument('--execute-sub-task', action='store_true', dest = 'execute_sub_task',
      help = argparse.SUPPRESS) #'Executes a subtask (FOR INTERNAL USE ONLY!!!)'
  parser.add_argument('--preprocess', action='store_true', 
      help = argparse.SUPPRESS) #'Perform VAD on the given range of audio files'
  parser.add_argument('--feature-extraction-training', action='store_true', dest = 'feature_extraction_training',
      help = argparse.SUPPRESS) #'Perform feature extraction for the given range of preprocessed audiofiles'
  parser.add_argument('--feature-extraction', action='store_true', dest = 'feature_extraction',
      help = argparse.SUPPRESS) #'Perform feature extraction for the given range of preprocessed audio files'
  parser.add_argument('--train-projector', action='store_true', dest = 'train_projector',
      help = argparse.SUPPRESS) #'Perform feature extraction training'
  parser.add_argument('--feature-projection-ubm', action='store_true', dest = 'projection_ubm',
      help = argparse.SUPPRESS) #'Perform feature projection ubm'
  parser.add_argument('--train-enroler', action='store_true', dest = 'train_enroler',
      help = argparse.SUPPRESS) #'Perform enrolment training'
  parser.add_argument('--train-whitening-enroler', action='store_true', dest = 'train_whitening_enroler',
      help = argparse.SUPPRESS) #'Perform enrolment training'
  parser.add_argument('--feature-projection-ivector', action='store_true', dest = 'projection_ivector',
      help = argparse.SUPPRESS) #'Perform feature projection ivector'
  parser.add_argument('--whitening-ivector', action='store_true', dest = 'whitening_ivector',
      help = argparse.SUPPRESS) #'Perform ivector whitening'
  parser.add_argument('--lnorm-ivector', action='store_true', dest = 'lnorm_ivector',
      help = argparse.SUPPRESS) #'Perform ivector whitening'
  parser.add_argument('--lda-train-projector', action='store_true', dest = 'lda_train_projector',
      help = argparse.SUPPRESS) #'Perform enrolment training'
  parser.add_argument('--lda-project-ivector', action='store_true', dest = 'lda_project_ivector',
      help = argparse.SUPPRESS) #'Perform LDA projection'
  parser.add_argument('--wccn-train-projector', action='store_true', dest = 'wccn_train_projector',
      help = argparse.SUPPRESS) #'Perform enrolment training'
  parser.add_argument('--wccn-project-ivector', action='store_true', dest = 'wccn_project_ivector',
      help = argparse.SUPPRESS) #'Perform WCCN projection'
  parser.add_argument('--train-plda-enroler', action='store_true', dest = 'train_plda_enroler',
      help = argparse.SUPPRESS) #'Perform WCCN projection'
  parser.add_argument('--enrol-models', action='store_true', dest = 'enrol_models',
      help = argparse.SUPPRESS) #'Generate the given range of models from the features'
  parser.add_argument('--model-type', type = str, choices = ['N', 'T'], metavar = 'TYPE', 
      help = argparse.SUPPRESS) #'Which type of models to generate (Normal or TModels)'
  parser.add_argument('--compute-scores', action='store_true', dest = 'compute_scores',
      help = argparse.SUPPRESS) #'Compute scores for the given range of models'
  parser.add_argument('--score-type', type = str, choices=['A', 'B', 'C', 'D', 'Z'],  metavar = 'SCORE', 
      help = argparse.SUPPRESS) #'The type of scores that should be computed'
  parser.add_argument('--group', type = str,  metavar = 'GROUP', 
      help = argparse.SUPPRESS) #'The group for which the current action should be performed'
  parser.add_argument('--concatenate', action='store_true',
      help = argparse.SUPPRESS) #'Concatenates the results of all scores of the given group'
  
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
  executor = ToolChainExecutorIVector(args)
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

