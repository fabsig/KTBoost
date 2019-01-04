# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:40:56 2018

@author: whsigris
"""

import numpy as np

class TesteEstimator(object):
    """An estimator predicting the alpha-quantile of the training targets."""
    def __init__(self, alpha=0.9):
        if not 0 < alpha < 1.0:
            raise ValueError("`alpha` must be in (0, 1.0) but was %r" % alpha)
        self.alpha = alpha

    def predict(self, X):
        print(np.ones(3))
        return self.alpha