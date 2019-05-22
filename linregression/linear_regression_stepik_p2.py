#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 23:30:56 2019

@author: valery.yakovlev
"""
import urllib
from urllib import request

import numpy as np
import pandas as pd

f = urllib.request.urlopen('https://stepik.org/media/attachments/lesson/16462/boston_houses.csv')  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with
my_df = pd.DataFrame(data)
my_df.head()

y = my_df[0]
X = my_df.drop([0], axis=1)

ones_col = np.ones(shape=(X.shape[0], 1))
X.insert(0, column='intercept', value=ones_col)

X_t = X.transpose()
dot_prod = X_t.dot(X)
df_inv = pd.DataFrame(np.linalg.pinv(dot_prod.values), dot_prod.columns, dot_prod.index)
one_more_dot = df_inv.dot(X_t)
beta = one_more_dot.dot(y)

for items in beta.iteritems():
    print(items[1])
