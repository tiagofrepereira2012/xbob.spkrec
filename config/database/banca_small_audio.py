#!/usr/bin/env python

import xbob.db.faceverif_fl

# 0/ The database to use
name = 'banca_small_audio'
db = xbob.db.faceverif_fl.Database('/idiap/user/ekhoury/LOBI/work/spkRecTool_2013_01_10/databases/banca/banca_small_audio/')
protocol = 'P'

img_input_dir = "/idiap/temp/ekhoury/databases/banca/wav_from_johnny/"
img_input_ext = ".wav"


