#!/usr/bin/env python

import xbob.db.voxforge

# 0/ The database to use
name = 'VoxForge'
# put the full path to the database filelist
db = xbob.db.voxforge.Database()
protocol = None

# put the full path to the folder that contains the audio files
wav_input_dir = "/idiap/temp/ekhoury/VoxForge/"
wav_input_ext = ".wav"

