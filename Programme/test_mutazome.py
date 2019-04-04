#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:22:47 2019

@author: routhier
"""

import numpy as np

from mutazome import _makebatch

def test_makebatch():
    nucleotid = np.array([[1], [1], [2], [3], [2], [4], [4], [1]])
    x_batch = _makebatch(nucleotid, 2, 4, 3)
    
    assert x_batch.shape == (3, 3, 4, 1) and x_batch[0, 2, 3, 0] == 1

