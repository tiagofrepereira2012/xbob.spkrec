#!/usr/bin/env python

import xbob.db.faceverif_fl

# 0/ The database to use
name = 'timit'
db = xbob.db.faceverif_fl.Database('/idiap/user/ekhoury/LOBI/work/spkRecTool_2013_01_10/databases/timit/2/')
protocol = '2'

img_input_dir = "/idiap/resource/database/timit/timit/"
img_input_ext = ".wav"

