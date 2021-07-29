# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:30:52 2021

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt

x = [(i+1)*10 for i in range(18)]
#print(x)

run1 = [0, 1399, 1676, 1464, 1674, 1612, 1528, 1644, 1110, 759, 792, 476, 397, 260, 161, 83, 31, 10]
run2 = [0, 1384, 1887, 1710, 1459, 1556, 1735, 1448, 1349, 787, 417, 313, 251, 155, 66, 31, 31, 31]
run3 = [0, 1356, 1760, 1879, 1680, 1751, 1639, 1545, 1294, 797, 553, 530, 333, 215, 131, 87, 60, 21]
run4 = [0, 1340, 1758, 1698, 1622, 1563, 1632, 1334, 1284, 1038, 654, 542, 468, 238, 83, 43, 28, 17]
run5 = [0, 1509, 1982, 1911, 1776, 1943, 1881, 1814, 1501, 956, 651, 393, 185, 77, 18, 5, 5, 0]

buffer_avg = []

for i in range(18) :
    avg = (run1[i] + run2[i] +run3[i] + run4[i] +run5[i])/5
    buffer_avg.append(avg)

print(buffer_avg)

plt.plot(x, buffer_avg , color = 'blue', label = "motif")
plt.legend()
plt.xlabel("time")
plt.ylabel("buffer consumed")
plt.show()