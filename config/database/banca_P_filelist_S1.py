#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'banca_p1'
db = xbob.db.verification.filelist.Database('/idiap/user/ekhoury/LOBI/work/spkRecLib2/databases/banca/P1')
protocol = 'P'

wav_input_dir = "/idiap/temp/ekhoury/databases/banca/wav_from_johnny/"
wav_input_ext = ".wav"

