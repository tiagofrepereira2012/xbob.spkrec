#!/usr/bin/env python

# setup of the grid parameters

# default queue used for training
training_queue = { 'queue':'q1dm', 'memfree':'64G', 'pe_opt':'pe_mth 8', 'hvmem':'8G' }
#training_queue = {'queue':'q1d', 'hostname':'*hx5*'} #{ 'queue':'q1d', 'memfree':'1G'}
# number of images that one job should preprocess
number_of_images_per_job = 500
preprocessing_queue = {'queue':'q1d'} #{'queue':'q1d', 'memfree':'1G'}

# number of features that one job should extract
number_of_features_per_job = 500
extraction_queue = {'queue':'q1d'}

# number of features that one job should project
number_of_projections_per_job = 500
projection_queue = {'queue':'q1d'}

# number of models that should be enroled by one enrol job
number_of_models_per_enrol_job = 20
enrol_queue = {'queue':'q1d'} #{ 'queue':'q1d', 'memfree':'1G' }

# number of models that one score computation should use
number_of_models_per_score_job = 50
#score_queue = {'queue':'q1d', 'hostname':'*hx5*'} #{ 'queue':'q1d', 'memfree':'1G' }

score_queue = {'queue':'q1d', 'io_big':True}
