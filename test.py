import numpy as np
from urllib.request import urlopen

response = urlopen('https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')

txt = np.loadtxt(response, skiprows=1, delimiter=',')

#
# avgs = [0]*7
# means = [0]*7
#
#
# row_cnt = txt.shape[0]-1
#
# for row in txt:
#     col_idx = 0
#     for col in row:
#         avgs[col_idx] += col
#         col_idx += 1
#
#
# mean_cnt = 0
#
# for _ in avgs:
#     means[mean_cnt] = _/row_cnt
#     mean_cnt+=1
#
# print(means)

#a = np.mean(txt, axis=0))

#print(a)
