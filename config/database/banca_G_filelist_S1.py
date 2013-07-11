#!/usr/bin/env python

import xbob.db.faceverif_fl

# 0/ The database to use
name = 'banca_G'
db = xbob.db.faceverif_fl.Database('databases/banca/G')
protocol = 'G'

img_input_dir = "/idiap/temp/ekhoury/databases/banca/wav_from_johnny/"
img_input_ext = ".wav"
pos_input_dir = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/"
pos_input_ext = ".pos"

annotation_type = 'eyecenter'

