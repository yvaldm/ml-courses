#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:09:19 2019

@author: valery.yakovlev
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

my_data = pd.read_csv('out.csv')

renamed_df = my_data.rename(index=str, columns={"Вес в килограммах": "key", "7.3": "value"})
renamed_df.head()
counts = renamed_df.groupby('key').agg('count')
counts.head()
counts.sort_values(by='value')

tfidf_transformer = TfidfTransformer()
count_vect = CountVectorizer()

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='log',
                                                   penalty='l2',
                                                   alpha=1e-3,
                                                   random_state=42))])

X = renamed_df['value']
y = renamed_df['key']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('training')
text_clf_svm = text_clf_svm.fit(X_train, y_train)
print('training finished')

print('dumping trained model to file')
pickle.dump(text_clf_svm, open('sgd_attribute_extraction_log.model', 'wb'))

res = text_clf_svm.predict_proba(X_test)
predicted_svm = text_clf_svm.predict(X_test)
np.mean(predicted_svm == y_test)
