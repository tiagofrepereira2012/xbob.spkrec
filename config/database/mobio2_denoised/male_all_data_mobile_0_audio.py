#!/usr/bin/env python

import xbob.db.mobio2

# setup for MoBio database
name = 'mobile0-male_all'
db = xbob.db.mobio2.Database()
protocol = 'mobile0-male'
world_gender = 'male' # gender dependent Enroler training
wav_input_dir = "/idiap/temp/ekhoury/MOBIO_DATABASE/denoisedAUDIO_16k/"
wav_input_ext = ".wav"

