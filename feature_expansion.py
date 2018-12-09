#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


# This fonction is used to expand the X data
def feature_expansion(data_array):
    
    N = data_array.shape[0]
    # dimension is the number of dimensions we have to work with
    dimension = data_array.shape[1]
    
    new_dimension = dimension*2 + dimension*(dimension - 1)
    
    # We will add first all the squares
    expanded_data = np.zeros((N, new_dimension))
    
    
    # We iterate through all the data points in order to extend the features of those points
    for point in range(N):
        
        # This is the current data point of length "dimension"
        current_point = data_array[point]
        
        expanded_data[point,0:dimension] = current_point
        expanded_data[point, dimension:dimension*2] = np.square(current_point)
        
        index = dimension*2
        
        for i in range(dimension):
            for j in range(dimension):
                if(i != j):
                    expanded_data[point, index] = current_point[i]*current_point[j]
                    index = index + 1
                    
    
    return expanded_data
    
    
    
    
    
    