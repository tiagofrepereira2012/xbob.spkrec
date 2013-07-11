#!/usr/bin/env python

import xbob.db.faceverif_fl

# 0/ The database to use
name = 'banca_p1'
db = xbob.db.faceverif_fl.Database('/idiap/user/ekhoury/LOBI/work/spkRecLib2/databases/banca/P1')
protocol = 'P'

img_input_dir = "/idiap/temp/ekhoury/databases/banca/wav_from_johnny/"
img_input_ext = ".wav"
pos_input_dir = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/"
pos_input_ext = ".pos"

annotation_type = 'eyecenter'

