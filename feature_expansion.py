#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# This fonction is used to expand the X data
def feature_expansion(X):
    X_expanded = np.c_[X, np.stack([np.multiply(x_i, x_j).T
                               for (x_i, x_j) in zip(X.T, X.T)], 1)]
    
    return X_expanded
