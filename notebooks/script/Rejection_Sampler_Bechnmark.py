#!/usr/bin/env python
# coding: utf-8

# Automatically reloads modules before executing code.
# This ensures that any changes made to imported Python files (e.g. .py modules) 
# are reflected in the notebook without needing to restart the kernel.

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from samplers.rejection_sampler import *


def 

