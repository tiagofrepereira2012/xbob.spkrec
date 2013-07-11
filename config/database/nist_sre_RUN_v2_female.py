#!/usr/bin/env python

import xbob.db.faceverif_fl

# 0/ The database to use
name = 'RUN_v2'
db = xbob.db.faceverif_fl.Database('/idiap/user/ekhoury/LOBI/work/spkRecLib5/databases/RUN_V2/female/')
protocol = 'F'

img_input_dir = "/idiap/temp/ekhoury/SRE2012/work/I4U/IDIAP/DATA/"
img_input_ext = ".sph"

