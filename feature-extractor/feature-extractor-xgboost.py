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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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
                         ('xgb', XGBClassifier(verbosity=2))])

X = renamed_df['value']
y = renamed_df['key']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('started training your xgboost')
text_clf_svm = text_clf_svm.fit(X_train, y_train)
print('finished training your xgboost')

print('dumping trained model to file')
pickle.dump(text_clf_svm, open('xgboost_attribute_extraction.model', 'wb'))

predicted_svm = text_clf_svm.predict(X_test)
np.mean(predicted_svm == y_test)

text = 'Оставайтесь на связи с малышом и днем, и ночью с видеоняней Philips Avent "SCD620/52". Система инфракрасного ночного видения автоматически включается с наступлением темноты и обеспечивает непревзойденную четкость. Вы можете наблюдать за спящим ребенком благодаря широкому цветному экрану 2,7" с высоким разрешением. Вы услышите даже тихий смех, малейший шорох и звук икоты с идеальной четкостью. Кристально чистое звучание позволяет не только наблюдать за малышом, но и слышать его в любое время. Включите режим ECO для снижения энергопотребления, когда малыш отдыхает. Уникальный индикатор соединения позволит следить за статусом подключения к детскому блоку. В режиме ECO отключается передача аудио и видео, а соединение между блоками восстанавливается при обнаружении звука в комнате малыша. Вы всегда будете знать о состоянии подключения и рабочем диапазоне видеоняни. Если прибор находится за пределами рабочего диапазона или разряжен, родительский блок оповестит вас об этом, чтобы вы всегда были на связи с малышом. Удобный родительский блок с беспроводной связью, не требующий подзарядки в течение 10 часов, обеспечивает свободу перемещений. '

for _ in text.split():
    print(_, '->', text_clf_svm.predict([_]))
