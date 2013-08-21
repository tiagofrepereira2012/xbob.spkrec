#!/usr/bin/env python

import xbob.db.mobio2

# setup for MoBio database
name = 'mobile0-female'
db = xbob.db.mobio2.Database()
protocol = 'mobile0-female'

wav_input_dir = "/idiap/temp/ekhoury/MOBIO_DATABASE/denoisedAUDIO_16k/"
wav_input_ext = ".wav"

world_projector_options = { 'subworld': "twothirds" }

