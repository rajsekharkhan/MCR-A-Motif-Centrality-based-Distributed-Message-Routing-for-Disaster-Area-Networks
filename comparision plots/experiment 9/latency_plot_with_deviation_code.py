# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 10:51:33 2021

@author: HP
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({'font.size': 18, 'font.weight': 'bold'})
labels = ['MCR', 'Flooding', 'PROPHET', 'WRP']

m_lat = [0.7720965506728918,
0.8537859007832899,
0.6852367688022284]

f_lat = [0.5121365185815456,
0.6863636363636364,
0.34156378600823045]

p_lat = [0.8855369799691832,
1.1866666666666668,
0.5833333333333334]

s_lat = [0.8618822524493618,
1.1128318584070795,
0.7108108108108108 ]

w = 0.20  # the width of the bars

model_latency_avg = [m_lat[0], f_lat[0], p_lat[0], s_lat[0]]
model_latency_max = [m_lat[1], f_lat[1], p_lat[1], s_lat[1]]
model_latency_min = [m_lat[2], f_lat[2], p_lat[2], s_lat[2]]

df = pd.DataFrame({'labels' : labels, "Time" :model_latency_avg, 'Min': model_latency_min, 'Max': model_latency_max})

# create ymin and ymax
df['ymin'] = df.Time - df.Min
df['ymax'] = df.Max - df.Time

# extract ymin and ymax into a (2, N) array as required by the yerr parameter
yerr = df[['ymin', 'ymax']].T.to_numpy()

# plot with error bars
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='labels', y='Time', data=df, yerr=yerr, ax=ax)
