#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'nist_male_small'
db = xbob.db.verification.filelist.Database('/idiap/user/ekhoury/LOBI/work/spkRecLib5/databases/nist_small/male/')
protocol = 'M'

wav_input_dir = "/idiap/temp/ekhoury/SRE2012/work/I4U/IDIAP/DATA/"
wav_input_ext = ".sph"

