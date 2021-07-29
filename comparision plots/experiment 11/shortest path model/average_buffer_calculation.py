# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:30:52 2021

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt

x = [(i+1)*10 for i in range(18)]
#print(x)

run1 = [0, 1740, 2612, 2927, 3070, 3320, 3657, 3816, 3872, 3735, 3243, 3130, 3289, 3124, 2944, 2666, 2463, 2108]
run2 = [0, 1744, 2881, 3700, 3253, 3585, 3818, 4093, 4143, 3841, 4373, 4177, 4367, 4346, 4219, 4146, 3640, 3694]
run3 = [0, 1377, 2654, 3198, 3129, 3281, 3101, 3560, 3378, 3659, 3800, 3399, 2767, 2512, 2201, 1685, 1222, 956]
run4 = [0, 1346, 2809, 3632, 3630, 3198, 3348, 3510, 3562, 3683, 3836, 3497, 3410, 3303, 2539, 1970, 1673, 1204]
run5 = [0, 1491, 3084, 3258, 3446, 3484, 3500, 3826, 3930, 3897, 3901, 3768, 3715, 3503, 3199, 3243, 3095, 3152]
 
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