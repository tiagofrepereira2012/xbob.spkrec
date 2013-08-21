#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'banca_small_audio'
db = xbob.db.verification.filelist.Database('/idiap/user/ekhoury/LOBI/work/databases/banca/banca_small_audio/')
protocol = 'P'

wav_input_dir = "/idiap/temp/ekhoury/databases/banca/wav_from_johnny/"
wav_input_ext = ".wav"


