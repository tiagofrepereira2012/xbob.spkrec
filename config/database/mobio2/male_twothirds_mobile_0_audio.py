#!/usr/bin/env python

import xbob.db.mobio2

# setup for MoBio database
name = 'mobile0-male'
db = xbob.db.mobio2.Database()
protocol = 'mobile0-male'

img_input_dir = "/idiap/temp/ekhoury/MOBIO_DATABASE/AUDIO/"
img_input_ext = ".wav"

world_projector_options = { 'subworld': "twothirds" }

