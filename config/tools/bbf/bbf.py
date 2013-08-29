#!/usr/bin/env python

import spkrec

tool = spkrec.tools.BBFTool

# 4/ BBF Enrolment and scoring
boosting_iterations = 500  # The number of iterations used in the discrete Ada Boosting
max_number_samples = 5000 # The maximum number of samples randomly choosen while boosting
