#!/usr/bin/env python

import spkrectool
import bob

tool = spkrectool.tools.LGBPHSTool

# distance function
distance_function = bob.math.histogram_intersection
is_distance_function = False

