#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'mobio_and_voxforge'
db = xbob.db.verification.filelist.Database('/idiap/user/ekhoury/LOBI/work/spkRecTool_2013_01_10/databases/mobio_and_voxforge/F/')
protocol = 'FEMALE'

wav_input_dir = "/idiap/temp/ekhoury/"
wav_input_ext = ".wav"

