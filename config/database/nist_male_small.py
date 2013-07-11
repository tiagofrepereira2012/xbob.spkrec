#!/usr/bin/env python

import xbob.db.faceverif_fl

# 0/ The database to use
name = 'nist_male_small'
db = xbob.db.faceverif_fl.Database('/idiap/user/ekhoury/LOBI/work/spkRecLib5/databases/nist_small/male/')
protocol = 'M'

img_input_dir = "/idiap/temp/ekhoury/SRE2012/work/I4U/IDIAP/DATA/"
img_input_ext = ".sph"
#pos_input_dir = "/idiap/group/vision/visidiap/databases/groundtruth/banca/english/eyecenter/"
#pos_input_ext = ".pos"

