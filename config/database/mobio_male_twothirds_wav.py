#!/usr/bin/env python

import xbob.db.mobio

# setup for MoBio database
name = 'mobio_2_3'
db = xbob.db.mobio.Database()
protocol = 'male'

img_input_dir = "/idiap/temp/ekhoury/databases/MOBIO/denoisedDATA_16k/"
img_input_ext = ".sph"
#img_input_dir = "/idiap/temp/pmotlic/TEMP.2/MOBIO/DATA"
#img_input_ext = ".wav"

world_projector_options = { 'subworld': "twothirds" }

