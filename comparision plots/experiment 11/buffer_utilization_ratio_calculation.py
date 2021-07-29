# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 21:24:39 2021

@author: HP
"""

buffer = [0, 1488, 3170, 3512, 3758, 3987, 3676, 3470, 3642, 3937, 3969, 4042, 3644, 3218, 3147, 3032, 2676, 2317]
alive_nodes = [50, 50, 50, 50, 50, 50, 47, 47, 47, 47, 46, 43, 36, 35, 32, 27, 25, 23]
buffer_utilization_ratio = []

for i in range(18) :
    ratio  = (buffer[i] / alive_nodes[i])
    buffer_utilization_ratio.append(ratio)

print(buffer_utilization_ratio)