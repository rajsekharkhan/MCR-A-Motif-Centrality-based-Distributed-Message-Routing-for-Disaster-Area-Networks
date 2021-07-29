# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 21:35:15 2021

@author: HP
"""


import numpy as np
import matplotlib.pyplot as plt

x = [(i+1)*10 for i in range(18)]
#print(x

motif = [0.0, 1397.6, 1812.6, 1732.4, 1642.2, 1685.0, 1683.0, 1557.0, 1307.6, 867.4, 613.4, 450.8, 326.8, 189.0, 91.8, 49.8, 31.0, 15.8]
shortest_path = [0.0, 1539.6, 2808.0, 3343.0, 3305.6, 3373.6, 3484.8, 3761.0, 3777.0, 3763.0, 3830.6, 3594.2, 3509.6, 3357.6, 3020.4, 2742.0, 2418.6, 2222.8]
flooding = [0.0, 9974.4, 968.2, 215.4, 148.2, 205.0, 154.0, 80.6, 55.2, 105.2, 105.2, 66.8, 25.4, 74.0, 30.0, 20.2, 0.8, 0.6]
prophet = [0.0, 949.4, 1938.0, 1999.2, 921.4, 1062.4, 1119.6, 841.2, 799.6, 844.6, 540.2, 412.2, 437.0, 356.6, 258.4, 239.6, 229.4, 171.2]

plt.plot(x, motif , color = 'blue', label = "MCR")
plt.plot(x, shortest_path , color = "green", label = "WRP")
plt.plot(x, flooding , color = "red", label = "flooding")
plt.plot(x, prophet , color = "black", label = "prophet")

plt.legend()
plt.xlabel("time")
plt.ylabel("buffer consumed")
plt.show()
