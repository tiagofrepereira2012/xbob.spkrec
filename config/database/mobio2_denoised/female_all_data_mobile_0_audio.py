#!/usr/bin/env python

import xbob.db.mobio2

# setup for MoBio database
name = 'mobile0-female_all'
db = xbob.db.mobio2.Database()
protocol = 'mobile0-female'
world_gender='female' # gender dependent Enroler training

img_input_dir = "/idiap/temp/ekhoury/MOBIO_DATABASE/denoisedAUDIO_16k/"
img_input_ext = ".wav"

