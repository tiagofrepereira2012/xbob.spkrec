#!/usr/bin/env python

import xbob.db.faceverif_fl

# 0/ The database to use
name = 'mobio_and_voxforge'
db = xbob.db.faceverif_fl.Database('/idiap/user/ekhoury/LOBI/work/spkRecTool_2013_01_10/databases/mobio_and_voxforge/M/')
protocol = 'MALE'

img_input_dir = "/idiap/temp/ekhoury/"
img_input_ext = ".wav"

