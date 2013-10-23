#!/usr/bin/env python

# setup of the grid parameters

# default queue used for training
training_queue = { 'queue':'q1dm', 'memfree':'16G', 'pe_opt':'pe_mth 2', 'hvmem':'8G', 'io_big':True }

# the queue that is used solely for the final ISV training step
isv_training_queue = { 'queue':'q1wm', 'memfree':'32G', 'pe_opt':'pe_mth 4', 'hvmem':'8G' }

# number of audio files that one job should preprocess
number_of_audio_files_per_job = 1000
preprocessing_queue = {}

# number of features that one job should extract
number_of_features_per_job = 600
extraction_queue = { 'queue':'q1d', 'memfree':'8G' }

# number of features that one job should project
number_of_projections_per_job = 600
projection_queue = { 'queue':'q1d', 'hvmem':'8G', 'memfree':'8G' }

# number of models that one job should enroll
number_of_models_per_enrol_job = 20
enrol_queue = { 'queue':'q1d', 'memfree':'4G', 'io_big':True }

# number of models that one score job should process
number_of_models_per_score_job = 20
score_queue = { 'queue':'q1d', 'memfree':'4G', 'io_big':True }

grid_type = 'sge' # on Idiap grid
