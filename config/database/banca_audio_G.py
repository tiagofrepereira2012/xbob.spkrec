#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'banca'
db = xbob.db.verification.filelist.Database('protocols/banca/')
protocol = 'G'

wav_input_dir = "/idiap/temp/ekhoury/databases/banca/wav/"
wav_input_ext = ".wav"

