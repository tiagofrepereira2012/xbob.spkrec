#!/usr/bin/env python

import spkrectool

tool = spkrectool.tools.BBFTool

# 4/ BBF Enrolment and scoring
boosting_iterations = 5  # The number of iterations used in the discrete Ada Boosting
max_number_samples = 4000 # The maximum number of samples randomly choosen while boosting
