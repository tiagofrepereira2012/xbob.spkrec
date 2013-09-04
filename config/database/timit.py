#!/usr/bin/env python

import xbob.db.verification.filelist

# 0/ The database to use
name = 'timit'
db = xbob.db.verification.filelist.Database('protocols/timit/2/')
protocol = None

# directory where the wave files are stored
wav_input_dir = "/path/to/timit/"
wav_input_ext = ".wav"

