#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'banca_G'
db = xbob.db.verification.filelist.Database('/idiap/user/ekhoury/LOBI/work/databases/banca/G/')
protocol = 'G'

img_input_dir = "/idiap/temp/ekhoury/databases/banca/wav_from_johnny/"
img_input_ext = ".wav"

