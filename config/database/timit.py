#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'timit'
db = xbob.db.verification.filelist.Database('/idiap/user/ekhoury/LOBI/work/spkRecTool_2013_01_10/databases/timit/2/')
protocol = '2'

wav_input_dir = "/idiap/resource/database/timit/timit/"
wav_input_ext = ".wav"

