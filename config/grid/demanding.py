#!/usr/bin/env python

# setup of the grid parameters

# default queue used for training
training_queue = { 'queue':'q1wm', 'memfree':'32G', 'pe_opt':'pe_mth 8', 'hvmem':'8G' }

# number of audio files that one job should preprocess
number_of_audio_files_per_job = 100
preprocessing_queue = {'queue':'q1d', 'memfree':'1G'}

# number of features that one job should extract
number_of_features_per_job = 100
extraction_queue = {'queue':'q1d', 'memfree':'1G'}

# number of features that one job should project
number_of_projections_per_job = 200
projection_queue = {'queue':'q1d', 'memfree':'1G'}

# number of models that should be enroled by one enrol job
number_of_models_per_enrol_job = 1
enrol_queue = { 'queue':'q1d', 'memfree':'1G' }

# number of models that one score computation should use
number_of_models_per_score_job = 1
score_queue = { 'queue':'q1d', 'memfree':'1G' }

grid_type = 'sge' # on Idiap grid
