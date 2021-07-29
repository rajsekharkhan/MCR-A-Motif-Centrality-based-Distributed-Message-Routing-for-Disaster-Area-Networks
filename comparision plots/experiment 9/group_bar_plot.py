# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:28:35 2021

@author: HP
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np



labels = ['high', 'medium', 'low']

model_pdr = [0.9696969696969697, 0.9178082191780822, 0.7978723404255319]
flooding_pdr = [0.45454545454545453, 0.43478260869565216, 0.42857142857142855]
shortest_path_pdr = [0.8823529411764706, 0.8714285714285714, 0.6875]

w = 0.20  # the width of the bars

x1 = np.arange(len(labels))  # the label locations
x2 = [i+w for i in x1]
x3 = [i+w for i in x2]



plt.bar(x2, model_pdr, w, label = "motif model pdr" )
plt.bar(x1, flooding_pdr, w, label = "flooding pdr")
plt.bar(x3, shortest_path_pdr, w, label = "Shortest path model pdr")
plt.ylabel('PDR')

plt.title('comparative pdr for model and flooding with same parameters')
plt.xticks(x1+w, labels)

plt.legend()

plt.show()