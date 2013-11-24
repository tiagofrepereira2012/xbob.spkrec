#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Manuel Guenther <Manuel.Guenther@idiap.ch>
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:49:02 CEST 2013
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
from .. import utils

class FileSelector:
  """This class provides shortcuts for selecting different files for different stages of the verification process"""
  
  def __init__(self, config, db):
    """Initialize the file selector object with the current configuration"""
    self.m_config = config
    self.m_db_options = db 
    self.m_db = db.db
    
  def __options__(self, name):
    """Returnes the options specified by the database, if available"""
    if hasattr(self.m_db_options, name):
      return eval('self.m_db_options.'+name)
    else:
      return {}
  
  def select_dir(self, dir_type):
    if dir_type == 'preprocessed':
      used_dir = self.m_config.preprocessed_dir 
    elif dir_type == 'features':
      used_dir = self.m_config.features_dir 
    elif dir_type == 'projected_ubm': 
      used_dir = self.m_config.projected_ubm_dir
    elif dir_type == 'projected_isv': 
      used_dir = self.m_config.projected_isv_dir
    elif dir_type == 'projected_ivector': 
      used_dir = self.m_config.projected_ivector_dir
    elif dir_type == 'whitened_ivector': 
      used_dir = self.m_config.whitened_ivector_dir
    elif dir_type == 'lnorm_ivector': 
      used_dir = self.m_config.lnorm_ivector_dir
    elif dir_type == 'lda_projected_ivector': 
      used_dir = self.m_config.lda_projected_ivector_dir
    elif dir_type == 'wccn_projected_ivector': 
      used_dir = self.m_config.wccn_projected_ivector_dir
    return used_dir  
  
  def select_all_files(self, directory, extension, tool_type):
    # default are the basic world model training files
    files = self.m_db.objects(protocol=self.m_config.protocol, **self.__options__('all_files_options'))
    if tool_type == 'ISV' or tool_type == 'IVector':
      if 'optional_world_1' in self.m_db.groups():
        files = files + self.m_db.objects(protocol=self.m_config.protocol, groups='optional_world_1', **self.__options__('all_files_options'))
    if tool_type == 'IVector':
      if 'optional_world_2' in self.m_db.groups():
        files = files + self.m_db.objects(protocol=self.m_config.protocol, groups='optional_world_2', **self.__options__('all_files_options'))    
    # sort files and remove duplicated files  
    files = self.sort(files)
    known = set()
    return [file.make_path(directory, extension) for file in files if file.path not in known and not known.add(file.path)]
  
  
  def select_training_files(self, group, directory, extension):
    # default are the basic world model training files
    files =  self.m_db.objects(protocol=self.m_config.protocol, groups=group, **self.__options__('all_files_options'))    
    # sort files and remove duplicated files  
    files = self.sort(files)
    known = set()
    return [file.make_path(directory, extension) for file in files if file.path not in known and not known.add(file.path)]

  
  def sort(self, files):
    """Returns a sorted version of the given list of File's (or other structures that define an 'id' data member).
    The files will be sorted according to their id, and duplicate entries will be removed."""
    sorted_files = sorted(files, cmp=lambda x,y: cmp(x.id, y.id))
    return [f for i,f in enumerate(sorted_files) if not i or sorted_files[i-1].id != f.id]  
    
    
  ### Original audio files and preprocessing
  def original_wav_list(self, tool_type):
    """Returns the list of original audio files that can be used for wav preprocessing"""
    directory=self.m_config.wav_input_dir
    extension=self.m_config.wav_input_ext
    return self.select_all_files(directory, extension, tool_type)
    
  def annotation_list(self, tool_type):
    """Returns the list of annotation files, if any (else None)"""
    if not hasattr(self.m_config, 'pos_input_dir') or self.m_config.pos_input_dir == None:
      return None
    directory=self.m_config.pos_input_dir
    extension=self.m_config.pos_input_ext
    return self.select_all_files(directory, extension, tool_type)
    
    
  def preprocessed_wav_list(self, tool_type):
    """Returns the list of preprocessed audio files and assures that the normalized wav path is existing"""
    utils.ensure_dir(self.m_config.preprocessed_dir)
    directory=self.m_config.preprocessed_dir
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)
    

  def feature_list(self, tool_type):
    """Returns the list of features and assures that the feature path is existing"""
    utils.ensure_dir(self.m_config.features_dir)
    directory=self.m_config.features_dir
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)

  ### Training and projecting features
  def training_wav_list(self):
    """Returns the list of audio files that should be used for extractor training"""
    directory=self.m_config.preprocessed_dir
    extension=self.m_config.default_extension
    return self.select_training_files('world', directory, extension)
    

  def training_feature_list(self):
    """Returns the list of features that should be used for projector training"""
    directory=self.m_config.features_dir
    extension=self.m_config.default_extension
    return self.select_training_files('world', directory, extension)
    
    
  def training_subspaces_list(self):
    """Returns the list of features that should be used for projector training"""
    directory=self.m_config.projected_ubm_dir
    extension=self.m_config.default_extension
    if 'optional_world_1' in self.m_db.groups():
      return self.select_training_files('optional_world_1', directory, extension)
    else:
      return self.select_training_files('world', directory, extension)

  def training_plda_list(self):
    """Returns the list of features that should be used for projector training"""
    directory=self.m_config.features_dir
    extension=self.m_config.default_extension
    if 'optional_world_2' in self.m_db.groups():
      return self.select_training_files('optional_world_2', directory, extension)
    else:
      return self.select_training_files('world', directory, extension)


  def training_feature_list_by_clients(self, dir_type, step):
    """Returns the list of training features, which is split up by the client ids."""
    # get the type of directory that is required
    cur_dir = self.select_dir(dir_type)
    # if requested, define the subset of training data to be used for this step
    if step == 'train_extractor':
      group = 'world'
      cur_world_options = self.__options__('world_extractor_options')
    elif step == 'train_projector':
      group = 'world'
      cur_world_options = self.__options__('world_projector_options')
    elif step == 'train_enroler':
      if 'optional_world_1' in self.m_db.groups():
        group = 'optional_world_1'
      else:
        group= 'world'
      cur_world_options = self.__options__('world_enroler_options')
    elif step == 'train_whitening_enroler':
      if 'optional_world_1' in self.m_db.groups():
        group = 'optional_world_1'
      else:
        group = 'world'
      cur_world_options = self.__options__('world_enroler_options')
    elif step == 'lda_train_projector':
      if 'optional_world_1' in self.m_db.groups():
        group = 'optional_world_1'
      else:
        group = 'world'
      cur_world_options = self.__options__('world_enroler_options')
    elif step == 'wccn_train_projector':
      if 'optional_world_1' in self.m_db.groups():
        group = 'optional_world_1'
      else:
        group = 'world'
      cur_world_options = self.__options__('world_enroler_options')
    elif step == 'train_plda_enroler':
      if 'optional_world_1' in self.m_db.groups():
        group = 'optional_world_2'
      else:
        group = 'world'
      cur_world_options = self.__options__('world_enroler_options')
      

    # iterate over all training clients
    features_by_clients_options = {}
    if 'subworld' in cur_world_options: features_by_clients_options['subworld'] = cur_world_options['subworld']
    features_by_clients_options.update(self.__options__('features_by_clients_options'))
    train_clients = self.m_db.clients(groups=group, protocol=self.m_config.protocol, **features_by_clients_options)
    training_filenames = {}
    for m in train_clients:
      # collect training features for current client id
      files = self.sort(self.m_db.objects(protocol=self.m_config.protocol, groups=group, model_ids=(m.id,), **cur_world_options))
      known = set()
      directory=cur_dir
      extension=self.m_config.default_extension
      train_data_m = [file.make_path(directory, extension) for file in files if file.path not in known and not known.add(file.path)]

      # add this model to the list
      training_filenames[m] = train_data_m
    # return the list of features which is grouped by client id
    return training_filenames
    

  def extractor_file(self):
    """Returns the file where to save the trainined extractor model to"""
    utils.ensure_dir(os.path.dirname(self.m_config.extractor_file))
    return self.m_config.extractor_file

  def projector_file(self):
    """Returns the file where to save the trained model"""
    utils.ensure_dir(os.path.dirname(self.m_config.projector_file))
    return self.m_config.projector_file
    
  def projected_list(self, dir_type, tool_type):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    directory = self.select_dir(dir_type)
    utils.ensure_dir(directory)
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)

  
  def projected_ubm_list(self, tool_type):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    utils.ensure_dir(self.m_config.projected_ubm_dir)
    directory=self.m_config.projected_ubm_dir
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)

  def projected_isv_list(self, tool_type='ISV'):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    utils.ensure_dir(self.m_config.projected_isv_dir)
    directory=self.m_config.projected_isv_dir
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)
      
  def projected_ivector_list(self, tool_type='IVector'):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    utils.ensure_dir(self.m_config.projected_ivector_dir)
    directory=self.m_config.projected_ivector_dir
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)

  def whitened_ivector_list(self, tool_type='IVector'):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    utils.ensure_dir(self.m_config.whitened_ivector_dir)
    
    directory=self.m_config.whitened_ivector_dir
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)

  def lnorm_ivector_list(self, tool_type='IVector'):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    utils.ensure_dir(self.m_config.lnorm_ivector_dir)
    directory=self.m_config.lnorm_ivector_dir
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)


  def lda_projected_ivector_list(self, tool_type='IVector'):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    utils.ensure_dir(self.m_config.lda_projected_ivector_dir)
    directory=self.m_config.lda_projected_ivector_dir
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)

  def wccn_projected_ivector_list(self, tool_type='IVector'):
    """Returns the list of projected features and assures that the projected feature path is existing"""
    utils.ensure_dir(self.m_config.wccn_projected_ivector_dir)
    directory=self.m_config.wccn_projected_ivector_dir
    extension=self.m_config.default_extension
    return self.select_all_files(directory, extension, tool_type)
    
    
  ### Enrolment
  def enroler_file(self):
    """Returns the name of the file that includes the model trained for enrolment"""
    utils.ensure_dir(os.path.dirname(self.m_config.enroler_file))
    return self.m_config.enroler_file
    
  ### Whitening Enrolment
  def whitening_enroler_file(self):
    """Returns the name of the file that includes the model trained for enrolment"""
    utils.ensure_dir(os.path.dirname(self.m_config.whitening_enroler_file))
    return self.m_config.whitening_enroler_file

  ### LDA Projection
  def lda_projector_file(self):
    """Returns the name of the file that includes the LDA projector"""
    utils.ensure_dir(os.path.dirname(self.m_config.lda_projector_file))
    return self.m_config.lda_projector_file
  
  
  ### WCCN Projection
  def wccn_projector_file(self):
    """Returns the name of the file that includes the WCCN projector"""
    utils.ensure_dir(os.path.dirname(self.m_config.wccn_projector_file))
    return self.m_config.wccn_projector_file
        
  ### PLDA Enroler
  def plda_enroler_file(self):
    """Returns the name of the file that includes the model trained for enrolment"""
    utils.ensure_dir(os.path.dirname(self.m_config.plda_enroler_file))
    return self.m_config.plda_enroler_file
        
  def model_ids(self, group):
    """Returns the sorted list of model ids from the given group"""
    return sorted(self.m_db.model_ids(groups=group, protocol=self.m_config.protocol))
  
  def client_id_from_model_id(self, model_id):
    """Returns the client id for the given model id."""
    if hasattr(self.m_db, 'get_client_id_from_model_id'):
      return self.m_db.get_client_id_from_model_id(model_id)
    else:
      return model_id
    
  def client_id(self, model_id):
    """Returns the id of the client for the given model id."""
    return self.client_id_from_model_id(model_id)
    
    
  def enrol_files(self, model_id, group, dir_type):
    """Returns the list of model features used for enrolment of the given model_id from the given group"""
    
    # get the type of directory that is required
    used_dir = self.select_dir(dir_type)
   
    files = self.sort(self.m_db.objects(groups=group, protocol=self.m_config.protocol, model_ids=(model_id,), purposes='enrol'))
    known = set()
    directory=used_dir
    extension=self.m_config.default_extension
    return [file.make_path(directory, extension) for file in files if file.path not in known and not known.add(file.path)]
    
  def model_files(self, model_id, group, dir_type):
    """Returns the files of the model and assures that the directory exists"""
    # get the type of directory that is required
    used_dir = self.select_dir(dir_type)
    model_file = os.path.join(self.m_config.models_dir, group, str(model_id) + self.m_config.default_extension) 
    utils.ensure_dir(os.path.dirname(model_file))
    files = self.sort(self.m_db.objects(groups=group, protocol=self.m_config.protocol, model_ids=(model_id,), purposes='enrol'))
    known = set()
    directory=used_dir
    extension=self.m_config.default_extension
    return [file.make_path(directory, extension) for file in files if file.path not in known and not known.add(file.path)]
  
  def model_file(self, model_id, group):
    """Returns the file of the model and assures that the directory exists"""
    model_file = os.path.join(self.m_config.models_dir, group, str(model_id) + self.m_config.default_extension) 
    utils.ensure_dir(os.path.dirname(model_file))
    return model_file  

  def tmodel_ids(self, group):
    """Returns the sorted list of T-Norm-model ids from the given group"""
    return sorted(self.m_db.tmodel_ids(groups=group))
    
  def tenrol_files(self, model_id, group, dir_type):
    """Returns the list of T-model features used for enrolment of the given model_id from the given group"""
    # get the type of directory that is required
    used_dir = self.select_dir(dir_type)
    tfiles = self.sort(self.m_db.tobjects(groups=group, model_ids=(model_id,),))
    known = set()
    directory=used_dir
    extension=self.m_config.default_extension
    return [file.make_path(directory, extension) for file in tfiles if file.path not in known and not known.add(file.path)]
    
    
  def tmodel_files(self, model_id, group, dir_type):
    """Returns the file of the T-Norm-model and assures that the directory exists"""
    used_dir = self.select_dir(dir_type)
    
    tmodel_file = os.path.join(self.m_config.tnorm_models_dir, group, str(model_id) + self.m_config.default_extension) 
    utils.ensure_dir(os.path.dirname(tmodel_file))

    files = self.sort(self.m_db.tobjects(groups=group, protocol=self.m_config.protocol, model_ids=(model_id,)))
    known = set()
    directory=used_dir
    extension=self.m_config.default_extension
    return [file.make_path(directory, extension) for file in files if file.path not in known and not known.add(file.path)]
  
  def tmodel_file(self, model_id, group):
    """Returns the file of the T-Norm-model and assures that the directory exists"""
    tmodel_file = os.path.join(self.m_config.tnorm_models_dir, group, str(model_id) + self.m_config.default_extension) 
    utils.ensure_dir(os.path.dirname(tmodel_file))
    return tmodel_file  

  ### Probe files and ZT-Normalization  
  def a_file(self, model_id, group):
    a_dir = os.path.join(self.m_config.zt_norm_A_dir, group)
    utils.ensure_dir(a_dir)
    return os.path.join(a_dir, str(model_id) + self.m_config.default_extension)

  def b_file(self, model_id, group):
    b_dir = os.path.join(self.m_config.zt_norm_B_dir, group)
    utils.ensure_dir(b_dir)
    return os.path.join(b_dir, str(model_id) + self.m_config.default_extension)

  def c_file(self, model_id, group):
    c_dir = os.path.join(self.m_config.zt_norm_C_dir, group)
    utils.ensure_dir(c_dir)
    return os.path.join(c_dir, "TM" + str(model_id) + self.m_config.default_extension)

  def c_file_for_model(self, model_id, group):
    c_dir = os.path.join(self.m_config.zt_norm_C_dir, group)
    return os.path.join(c_dir, str(model_id) + self.m_config.default_extension)
    
  def d_file(self, model_id, group):
    d_dir = os.path.join(self.m_config.zt_norm_D_dir, group)
    utils.ensure_dir(d_dir)
    return os.path.join(d_dir, str(model_id) + self.m_config.default_extension)
    
  def d_matrix_file(self, group):
    d_dir = os.path.join(self.m_config.zt_norm_D_dir, group)
    return os.path.join(d_dir, "D" + self.m_config.default_extension)
    
  def d_same_value_file(self, model_id, group):
    d_dir = os.path.join(self.m_config.zt_norm_D_sameValue_dir, group)
    utils.ensure_dir(d_dir)
    return os.path.join(d_dir, str(model_id) + self.m_config.default_extension)

  def d_same_value_matrix_file(self, group):
    d_dir = os.path.join(self.m_config.zt_norm_D_sameValue_dir, group)
    return os.path.join(d_dir, "D_sameValue" + self.m_config.default_extension)
  
  def no_norm_file(self, model_id, group):
    norm_dir = os.path.join(self.m_config.scores_nonorm_dir, group)
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, str(model_id) + ".txt")
    
  def no_norm_result_file(self, group):
    norm_dir = self.m_config.scores_nonorm_dir
    return os.path.join(norm_dir, "scores-" + group)
    

  def zt_norm_file(self, model_id, group):
    norm_dir = os.path.join(self.m_config.scores_ztnorm_dir, group)
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, str(model_id) + ".txt")
    
  def zt_norm_result_file(self, group):
    norm_dir = self.m_config.scores_ztnorm_dir
    utils.ensure_dir(norm_dir)
    return os.path.join(norm_dir, "scores-" + group)
  

  def probe_files(self, group, dir_type):
    """Returns the probe files used to compute the raw scores"""
    objects = self.sort(self.m_db.objects(groups=group, protocol=self.m_config.protocol, purposes="probe"))
    known = set()
    directory=self.select_dir(dir_type)
    extension=self.m_config.default_extension
    return [file.make_path(directory, extension) for file in objects if file.path not in known and not known.add(file.path)]
    
    
  def zprobe_files(self, group, dir_type):
    """Returns the probe files used to compute the Z-Norm"""
    objects = self.sort(self.m_db.zobjects(groups=group)) # This is the default
    known = set()
    directory=self.select_dir(dir_type)
    extension=self.m_config.default_extension
    return [file.make_path(directory, extension) for file in objects if file.path not in known and not known.add(file.path)]
    
    
  def probe_files_for_model(self, model_id, group, dir_type):
    """Returns the probe files used to compute the raw scores"""
    objects = self.sort(self.m_db.objects(groups=group, protocol=self.m_config.protocol, purposes="probe", model_ids=(model_id,)))
    known = set()
    directory=self.select_dir(dir_type)
    extension=self.m_config.default_extension
    return [file.make_path(directory, extension) for file in objects if file.path not in known and not known.add(file.path)]
    
  def zprobe_files_for_model(self, model_id, group, dir_type):
    """Returns the probe files used to compute the Z-Norm"""
    objects = self.sort(self.m_db.zobjects(protocol=self.m_config.protocol, groups=group, model_ids=(model_id,))) # This is the default
    known = set()
    directory=self.select_dir(dir_type)
    extension=self.m_config.default_extension
    return [file.make_path(directory, extension) for file in objects if file.path not in known and not known.add(file.path)]


  def probe_objects(self, group):
    """Returns the probe object used to compute the raw scores"""
    return self.sort(self.m_db.objects(groups=group, protocol=self.m_config.protocol, purposes="probe"))
    
    
  def zprobe_objects(self, group):
    """Returns the probe objects used to compute the Z-Norm"""
    return self.sort(self.m_db.zobjects(protocol=self.m_config.protocol, groups=group))  # This is the default
    

  def probe_objects_for_model(self, model_id, group):
    """Returns the probe objects used to compute the raw scores"""
    return self.sort(self.m_db.objects(groups=group, protocol=self.m_config.protocol, purposes="probe", model_ids=(model_id,)))
   
   
  def zprobe_objects_for_model(self, model_id, group):
    """Returns the probe objects used to compute the Z-Norm"""
    return self.sort(self.m_db.zobjects(protocol=self.m_config.protocol, groups=group, model_ids=(model_id,)))  # This is the default
    
    
