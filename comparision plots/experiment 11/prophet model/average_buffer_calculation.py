# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:30:52 2021

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt

x = [(i+1)*10 for i in range(18)]
#print(x)

run1 = [0, 623, 1593, 1533, 1362, 1345, 975, 1294, 827, 988, 622, 713, 555, 393, 228, 220, 422, 365]
run2 = [0, 793, 1979, 1640, 522, 992, 696, 683, 1021, 1133, 712, 212, 534, 399, 182, 89, 133, 93]
run3 = [0, 512, 1441, 2166, 1120, 1434, 718, 242, 241, 354, 408, 404, 418, 365, 295, 385, 261, 90]
run4 = [0, 1385, 2215, 2542, 751, 1015, 1156, 714, 734, 620, 530, 508, 458, 403, 389, 282, 181, 209]
run5 = [0, 1434, 2462, 2115, 852, 526, 2053, 1273, 1175, 1128, 429, 224, 220, 223, 198, 222, 150, 99] 

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