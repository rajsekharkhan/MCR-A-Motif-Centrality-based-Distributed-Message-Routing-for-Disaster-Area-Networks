# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 11:19:41 2021

@author: HP
"""

run_1 = [0.3333333333333333, 0.46511627906976744, 0.6197183098591549, 0.717391304347826, 0.7129629629629629, 0.7096774193548387, 0.6283783783783784, 0.5909090909090909, 0.6464646464646465, 0.6203703703703703, 0.5772357723577236, 0.5424354243542435, 0.5217391304347826, 0.4890282131661442, 0.45217391304347826, 0.42391304347826086, 0.3969465648854962, 0.3696682464454976]
run_2 = [0.2, 0.5263157894736842, 0.6521739130434783, 0.6530612244897959, 0.7131147540983607, 0.7083333333333334, 0.6909090909090909, 0.6451612903225806, 0.6273584905660378, 0.6042553191489362, 0.5576923076923077, 0.534965034965035, 0.4875, 0.4482758620689655, 0.41935483870967744, 0.39097744360902253, 0.36705882352941177, 0.3443708609271523]
run_3 = [0.5294117647058824, 0.5384615384615384, 0.6363636363636364, 0.7263157894736842, 0.8205128205128205, 0.8041958041958042, 0.757396449704142, 0.6923076923076923, 0.6743119266055045, 0.6378600823045267, 0.5851851851851851, 0.5608108108108109, 0.5220125786163522, 0.47564469914040114, 0.4450402144772118, 0.41089108910891087, 0.3905882352941176, 0.36807095343680707]
run_4 = [0.22727272727272727, 0.5434782608695652, 0.582089552238806, 0.7078651685393258, 0.7894736842105263, 0.7482014388489209, 0.7134146341463414, 0.7074468085106383, 0.6650717703349283, 0.6170212765957447, 0.58984375, 0.568904593639576, 0.5275080906148867, 0.49393939393939396, 0.4527777777777778, 0.42010309278350516, 0.4034653465346535, 0.37906976744186044]
run_5 = [0.2631578947368421, 0.5789473684210527, 0.7878787878787878, 0.7666666666666667, 0.7946428571428571, 0.8409090909090909, 0.7407407407407407, 0.7297297297297297, 0.6604651162790698, 0.6302521008403361, 0.6254826254826255, 0.583916083916084, 0.545751633986928, 0.5030120481927711, 0.4717514124293785, 0.4489247311827957, 0.42065491183879095, 0.39952153110047844]

avg_pdr = []
min_pdr = []
max_pdr = []

min_val = 100
max_val = -1

for i in range(18):
    """
    min_val = 100
    max_val = -1
    """
    all_pdr = [run_1[i], run_2[i], run_3[i], run_4[i], run_5[i]]
    sum_pdr = sum(all_pdr)
    avg_val = sum_pdr / len(all_pdr)
    avg_pdr.append(avg_val)
    max_val = max(all_pdr)
    max_pdr.append(max_val)
    min_val = min(all_pdr)
    min_pdr.append(min_val)
    
print(avg_pdr)
print(max_pdr)
print(min_pdr)
    