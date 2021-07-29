# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:44:05 2021

@author: HP
"""

latency = [0.7775,
0.7108108108108108,
1.1128318584070795,
0.9255583126550868,
0.7827102803738317]

max_lat = -1
min_lat = 1000

total = 0

for i in range(len(latency)):
    total += latency[i]
    
    if latency[i]>max_lat :
        max_lat = latency[i]
        
    if latency[i] < min_lat :
        min_lat = latency[i]
        
avg_latency = total/(len(latency))
print(avg_latency)
print(max_lat)
print(min_lat)