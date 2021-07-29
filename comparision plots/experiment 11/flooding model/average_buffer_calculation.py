# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:30:52 2021

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt

x = [(i+1)*10 for i in range(18)]
#print(x)

run1 = [0, 9482, 995, 28, 37, 32, 272, 68, 38, 342, 230, 252, 5, 6, 10, 1, 0, 0]
run2 = [0, 5677, 2109, 324, 101, 413, 91, 2, 11, 5, 52, 26, 10, 15, 54, 29, 0, 0]
run3 = [0, 10932, 911, 33, 44, 79, 25, 22, 60, 46, 12, 36, 94, 318, 6, 12, 2, 2]
run4 = [0, 15636, 338, 182, 125, 83, 93, 13, 24, 13, 126, 19, 18, 31, 80, 59, 2, 1]
run5 = [0, 8145, 488, 510, 434, 418, 289, 298, 143, 120, 106, 1, 0, 0, 0, 0, 0, 0]

buffer_avg = []

for i in range(18) :
    avg = (run1[i] + run2[i] +run3[i] + run4[i] +run5[i])/5
    buffer_avg.append(avg)

print(buffer_avg)

plt.plot(x, buffer_avg , color = 'blue', label = "flooding")
plt.legend()
plt.xlabel("time")
plt.ylabel("buffer consumed")
plt.show()