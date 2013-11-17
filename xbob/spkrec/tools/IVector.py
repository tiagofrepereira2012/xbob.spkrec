#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Elie Khoury <Elie.Khoury@idiap.ch>
# Fri Aug 30 11:53:40 CEST 2013
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

import bob
import numpy
from . import UBMGMMTool
from itertools import izip
from math import sqrt
import logging

logger = logging.getLogger("bob.c++")
logger.setLevel(logging.INFO)
logger.propagate = True

def cosine_distance(a, b):
    if len(a) != len(b):
        raise ValueError, "a and b must be same length"
    numerator = sum(tup[0] * tup[1] for tup in izip(a,b))
    denoma = sum(avalue ** 2 for avalue in a)
    denomb = sum(bvalue ** 2 for bvalue in b)
    result = numerator / (sqrt(denoma)*sqrt(denomb))
    return result
    

class IVecTool (UBMGMMTool):
  """Tool chain for computing Unified Background Models and Gaussian Mixture Models of the features"""

  
  def __init__(self, setup):
    """Initializes the local UBM-GMM tool with the given file selector object"""
    # call base class constructor
    UBMGMMTool.__init__(self, setup)
    del self.use_unprojected_features_for_model_enrol


  def __load_gmm_stats_list__(self, ld_files):
    """Loads a list of lists of GMM statistics from a list of dictionaries of filenames
       There is one list for each identity"""
    # Initializes a python list for the GMMStats
    gmm_stats = [] 
    for k in sorted(ld_files.keys()): 
      for f in ld_files[k]: 
        # Processes one file 
        stats = bob.machine.GMMStats( bob.io.HDF5File(str(f)) ) 
        # Appends in the list 
        gmm_stats.append(stats)
    return gmm_stats


  #######################################################
  ################ IVector training #########################
  def train_enroler(self, train_files, enroler_file):
  
    # load GMM stats from training files
    gmm_stats = self.__load_gmm_stats_list__(train_files)
    
    # create a IVectorMachine with the UBM from the base class
    self.m_ivector = bob.machine.IVectorMachine(self.m_ubm, self.m_config.rt) #This is the dimension of the T matrix. It is tipically equal to 400. 
    self.m_ivector.variance_threshold = 1e-5 
    
    t = numpy.random.randn(self.m_ubm.dim_c * self.m_ubm.dim_d, self.m_config.rt)
    sigma = self.m_ubm.variance_supervector
    self.m_ivector.t = t 
    self.m_ivector.sigma = sigma

    # Initialization of the IVectorTrainer
    trainer = bob.trainer.IVectorTrainer(update_sigma=True, convergence_threshold= self.m_config.convergence_threshold, max_iterations=self.m_config.max_iterations)
    trainer.initialize(self.m_ivector, gmm_stats)

    # E-Step
    trainer.train(self.m_ivector, gmm_stats)
    # M-Step
    #trainer.m_step(self.m_ivector, gmm_stats)

    # Save the i-vector machine base AND the UBM into the same file
    self.m_ivector.save(bob.io.HDF5File(enroler_file, "w"))

  
  #######################################################
  ################ Withening training #########################
  def train_whitening_enroler(self, train_files, whitening_enroler_file):
  
    # load GMM stats from training files

    ivectors_matrix  = []
    for k in sorted(train_files.keys()):
      for f in train_files[k]:
        ivec = (bob.io.HDF5File(f)).read('ivec')
        ivectors_matrix.append(ivec)
    ivectors_matrix = numpy.vstack(ivectors_matrix)
    
    # create a Linear Machine     # Runs whitening (first method)
    self.whitening_machine = bob.machine.LinearMachine(ivectors_matrix.shape[1],ivectors_matrix.shape[1])
    
    # create the whitening trainer
    t = bob.trainer.WhiteningTrainer()
    
    t.train(self.whitening_machine, ivectors_matrix)
    
    # Save the whitening linear machine
    print("Saving the whitening machine..")
    self.whitening_machine.save(bob.io.HDF5File(whitening_enroler_file, "w"))
   

    
  #######################################################
  ################ PLDA training #########################
  def __train_pca__(self, training_set):
    """Trains and returns a LinearMachine that is trained using PCA"""
    data_list = []
    for client in training_set:
      for feature in client:
        # Appends in the array
        data_list.append(feature)
    data = numpy.vstack(data_list)

    print("  -> Training LinearMachine using PCA (SVD)")
    t = bob.trainer.SVDPCATrainer()
    machine, __eig_vals = t.train(data)
    # limit number of pcs
    machine.resize(machine.shape[0], self.m_config.subspace_dimension_pca)
    return machine

  def __perform_pca_client__(self, machine, client):
    """Perform PCA on an array"""
    client_data_list = []
    for feature in client:
      # project data
      projected_feature = numpy.ndarray(machine.shape[1], numpy.float64)
      machine(feature, projected_feature)
      # add data in new array
      client_data_list.append(projected_feature)
    client_data = numpy.vstack(client_data_list)
    return client_data

  def __perform_pca__(self, machine, training_set):
    """Perform PCA on data"""
    data = []
    for client in training_set:
      client_data = self.__perform_pca_client__(machine, client)
      data.append(client_data)
    return data
    
  ########### PLDA training ############
  def train_plda_enroler(self, train_files, plda_enroler_file):
  
    # load GMM stats from training files
    training_features = self.load_ivectors_by_client(train_files)
    
    
    # train PCA and perform PCA on training data
    if self.m_config.subspace_dimension_pca is not None:
      self.m_pca_machine = self.__train_pca__(training_features)
      training_features = self.__perform_pca__(self.m_pca_machine, training_features)
    
    input_dimension = training_features[0].shape[1]

    print("  -> Training PLDA base machine")
    # create trainer
    t = bob.trainer.PLDATrainer(self.m_config.PLDA_TRAINING_ITERATIONS)
    
    t.seed = self.m_config.INIT_SEED
    
    t.init_f_method = self.m_config.INIT_F_METHOD
    t.init_f_ratio = self.m_config.INIT_F_RATIO
    t.init_g_method = self.m_config.INIT_G_METHOD
    t.init_g_ratio = self.m_config.INIT_G_RATIO
    t.init_sigma_method = self.m_config.INIT_S_METHOD
    t.init_sigma_ratio = self.m_config.INIT_S_RATIO

    # train machine
    self.m_plda_base = bob.machine.PLDABase(input_dimension, self.m_config.SUBSPACE_DIMENSION_OF_F, self.m_config.SUBSPACE_DIMENSION_OF_G, self.m_config.variance_flooring)
    t.train(self.m_plda_base, training_features)

    # write machines to file
    proj_hdf5file = bob.io.HDF5File(str(plda_enroler_file), "w")
    if self.m_config.subspace_dimension_pca is not None:
      proj_hdf5file.create_group('/pca')
      proj_hdf5file.cd('/pca')
      self.m_pca_machine.save(proj_hdf5file)
    proj_hdf5file.create_group('/plda')
    proj_hdf5file.cd('/plda')
    self.m_plda_base.save(proj_hdf5file)


  #######################################################
  ################## IVector model enrol ####################
  def load_enroler(self, enroler_file):
    """Reads the UBM model from file"""
    # now, load the JFA base, if it is included in the file

    self.m_ivector = bob.machine.IVectorMachine(self.m_ubm, self.m_config.rt)
    self.m_ivector.load(bob.io.HDF5File(enroler_file))
    #self.m_ivector = bob.machine.IVectorMachine(bob.io.HDF5File(enroler_file))

    # add UBM model from base class
    self.m_ivector.ubm = self.m_ubm


  #######################################################
  ############## Whitening model enrol ##################
  
  def load_whitening_enroler(self, whitening_enroler_file):
    """Reads the whitening Enroler model from file"""
    # now, load the JFA base, if it is included in the file
    
    self.whitening_machine = bob.machine.LinearMachine(self.m_config.rt,self.m_config.rt)
    self.whitening_machine.load(bob.io.HDF5File(whitening_enroler_file))
   
  #######################################################
  ############## PLDA model enrol ##################
  
  def load_plda_enroler(self, plda_enroler_file):
    """Reads the PCA projection matrix and the PLDA model from file"""
    # read UBM
    proj_hdf5file = bob.io.HDF5File(plda_enroler_file)
    if self.m_config.subspace_dimension_pca is not None:
      proj_hdf5file.cd('/pca')
      self.m_pca_machine = bob.machine.LinearMachine(proj_hdf5file)
    proj_hdf5file.cd('/plda')
    self.m_plda_base = bob.machine.PLDABase(proj_hdf5file)
    self.m_plda_machine = bob.machine.PLDAMachine(self.m_plda_base)
    self.m_plda_trainer = bob.trainer.PLDATrainer()
    
  def plda_enrol(self, enroll_features):
    """Enrolls the model by computing an average of the given input vectors"""
    enroll_features = numpy.vstack(enroll_features)
    if self.m_config.subspace_dimension_pca is not None:
      enroll_features_projected = self.__perform_pca_client__(self.m_pca_machine, enroll_features)
      self.m_plda_trainer.enrol(self.m_plda_machine,enroll_features_projected)
    else:
      self.m_plda_trainer.enrol(self.m_plda_machine,enroll_features)
    return self.m_plda_machine
 
  def read_plda_model(self, model_file):
    """Reads the model, which in this case is a PLDA-Machine"""
    # read machine and attach base machine
    print ("model: %s" %model_file)
    plda_machine = bob.machine.PLDAMachine(bob.io.HDF5File(str(model_file)), self.m_plda_base)
    return plda_machine
  
  def plda_score(self, model, probe):
    return model.forward(probe)
    
  def project_ivector(self, feature_array, projected_ubm):
    m_ivector = bob.machine.IVectorMachine(self.m_ivector)
    projected_ivector = m_ivector.forward(projected_ubm)
    return projected_ivector
  
  def save_feature(self, data, feature_file):
    hdf5file = bob.io.HDF5File(feature_file, "w")
    hdf5file.set('ivec', data)

  def read_feature(self, feature_file):
    """Reads the projected feature to be enroled as a model"""
    return bob.machine.GMMStats(bob.io.HDF5File(str(feature_file))) 
    
  def read_ivector(self, ivector_file):
    """Reads the ivectors that correspond to the model, and put them in a list"""
    return (bob.io.HDF5File(str(ivector_file))).read('ivec')


  def read_model(self, model_files):
    """Reads the ivectors that correspond to the model, and put them in a list"""
    return [(bob.io.HDF5File(k)).read('ivec') for k in model_files]


  def load_ivectors_by_client(self, ld_files):
    """Loads a list of lists of i-vectors
       There is one list for each identity"""
    ivectors = [] 
    for k in sorted(ld_files.keys()): 
      ivec_client = []
      for f in ld_files[k]:
        ivec = self.read_ivector(f)
        ivec_client.append(ivec)
      ivec_client = numpy.vstack(ivec_client)
      ivectors.append(ivec_client)
    return ivectors
    
      
  def read_probe(self, probe_file):
    """Read the type of features that we require, namely GMMStats"""
    hdf5file = bob.io.HDF5File(probe_file)
    ivec = hdf5file.read('ivec')
    return ivec
    

  ######################################################
  ################ Feature comparison ##################
  def read_ivectors(self, client_files):
    return numpy.vstack([self.read_ivector(f) for f in client_files])
    
  def cosine_score(self, client_ivectors, probe_ivector):
    """Computes the score for the given model and the given probe using the scoring function from the config file"""
    scores = []
    for ivec in client_ivectors:
      scores.append(cosine_distance(ivec, probe_ivector))
    return numpy.max(scores)
    
  def whitening_ivector(self, ivector):
    m = self.whitening_machine
    whited_ivector = m.forward(ivector)
    return whited_ivector

  def lnorm_ivector(self, ivector):
    return ivector/numpy.linalg.norm(ivector)
    
  def score_with_whitening(self, model, probe):
    m = self.whitening_machine
    probe_w = m.forward(probe)
    """Computes the score for the given model and the given probe using the scoring function from the config file"""
    scores = []
    for ivec in model:
      ivec_w = m.forward(ivec)
      scores.append(cosine_distance(ivec_w, probe_w))
    return numpy.max(scores)
    

    
   #######################################################
   ################## LDA projection #####################
  
  def lda_read_data(self, training_files):
    data = []
    for c in training_files:
      # at least two files per client are required!
      client_files=training_files[c]
      if len(client_files) < 2:
        print("Skipping one client since the number of client files is only %d" % len(client_files))
        continue
      data.append(numpy.vstack([self.read_ivector(f) for f in client_files]))

    # Returns the list of lists of arrays
    return data
      
  def lda_train_projector(self, training_files, lda_projector_file):
    """Generates the LDA projection matrix from the given features (that are sorted by identity)"""
    # Initializes an array for the data
    data = self.lda_read_data(training_files)
    print("  -> Training LinearMachine using LDA")
    #t = bob.trainer.FisherLDATrainer()
    # In case of trouble, use the pseudo-inverse computation flag to true
    #t = bob.trainer.FisherLDATrainer(use_pinv=True)
    t = bob.trainer.FisherLDATrainer(strip_to_rank=False)
    self.lda_machine, __eig_vals = t.train(data)
    # resize the machine if desired
    if self.m_config.LDA_SUBSPACE_DIMENSION:
      self.lda_machine.resize(self.lda_machine.shape[0], self.m_config.LDA_SUBSPACE_DIMENSION)
    self.lda_machine.save(bob.io.HDF5File(lda_projector_file, "w"))

  def lda_load_projector(self, lda_projector_file):
    """Reads the UBM model from file"""
    # read LDA projector
    self.lda_machine = bob.machine.LinearMachine(bob.io.HDF5File(lda_projector_file))
    # Allocates an array for the projected data
    self.m_projected_feature = numpy.ndarray(self.lda_machine.shape[1], numpy.float64)

  def lda_project_ivector(self, feature):
    """Projects the data using the stored covariance matrix"""
    # Projects the data
    self.lda_machine(feature, self.m_projected_feature)
    # return the projected data
    return self.m_projected_feature

  ##########################################################
  ################### WCCN Projection ######################
      
  def wccn_train_projector(self, training_files, wccn_projector_file):
    """Generates the WCCN projection matrix from the given features (that are sorted by identity)"""
    # Initializes an array for the data
    data = self.lda_read_data(training_files) # reading the data is the same as for LDA training
    print("  -> Training LinearMachine using WCCN")
    t = bob.trainer.WCCNTrainer()
    self.wccn_machine = t.train(data)
    self.wccn_machine.save(bob.io.HDF5File(wccn_projector_file, "w"))

  def wccn_load_projector(self, wccn_projector_file):
    """Reads the WCCN projector from file"""
    # read WCCN projector
    self.wccn_machine = bob.machine.LinearMachine(bob.io.HDF5File(wccn_projector_file))
    # Allocates an array for the projected data
    self.m_projected_feature = numpy.ndarray(self.wccn_machine.shape[1], numpy.float64)

  def wccn_project_ivector(self, feature):
    """Projects the data using the stored covariance matrix"""
    # Projects the data
    self.wccn_machine(feature, self.m_projected_feature)
    # return the projected data
    return self.m_projected_feature

