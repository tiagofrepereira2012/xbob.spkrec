#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'banca'
db = xbob.db.verification.filelist.Database('protocols/banca/')
protocol = 'G'

# directory where the wave files are stored
wav_input_dir = "/path/to/banca/"
wav_input_ext = ".wav"

