#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'RUN_v2'
db = xbob.db.verification.filelist.Database('/idiap/user/ekhoury/LOBI/work/spkRecLib5/databases/RUN_V2/female/')
protocol = 'F'

wav_input_dir = "/idiap/temp/ekhoury/SRE2012/work/I4U/IDIAP/DATA/"
wav_input_ext = ".sph"

