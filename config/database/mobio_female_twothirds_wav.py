#!/usr/bin/env python

import xbob.db.mobio

# setup for MoBio database
name = 'mobio_2_3'
db = xbob.db.mobio.Database()
protocol = 'female'

img_input_dir = "/idiap/temp/ekhoury/databases/MOBIO/denoisedDATA_16k/"
img_input_ext = ".sph"

world_projector_options = { 'subworld': "twothirds" }

