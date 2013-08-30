#!/usr/bin/env python

import xbob.spkrec

tool = xbob.spkrec.tools.BBFTool

# 4/ BBF Enrolment and scoring
boosting_iterations = 10  # The number of iterations used in the discrete Ada Boosting
max_number_samples = 4000 # The maximum number of samples randomly choosen while boosting
