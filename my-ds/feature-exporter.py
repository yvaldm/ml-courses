#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:09:19 2019

@author: valery.yakovlev
"""





import numpy as np
import pandas as pd


my_data = pd.read_csv('out.csv')


my_data.head()

renamed_df = my_data.rename(index=str, columns={"Вес в килограммах": "key", "7.3": "value"})
renamed_df.head()
counts = renamed_df.groupby('key').agg('count')
counts.head()
counts.sort_values(by='value')



from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
count_vect = CountVectorizer()

text_clf_svm = Pipeline([('vect', CountVectorizer()), 
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', 
                                                   penalty='l2', 
                                                   alpha=1e-3, 
                                                   n_iter=5, 
                                                   random_state=42)),])

X = renamed_df['value']
y = renamed_df['key']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

text_clf_svm = text_clf_svm.fit(X_train, y_train)
predicted_svm = text_clf_svm.predict(X_test)
np.mean(predicted_svm == y_test)

text = 'geeks for geeks'
print(text.split()) 

text = 'Wi-Fi-Видеоняня Samsung SmartCam SNH-V6410PNW представляет собой камеру с широким устойчивым основанием, которая позволит круглосуточно мониторить детскую комнату, следя за поведением ребенка и мгновенно отправляя данные о движении или плаче на подключенный к камере гаджет. Таким образом наблюдение за комнатой малыша можно не прерывать, где бы вы не находились, необходимым условием является только наличие интернета на устройстве, к которому подключена видеоняня Samsung SNH-V6410PNW. ОСОБЕННОСТИ: роль родительского блока может выполнять смартфон, планшет или ноутбук, позволяющие удаленно управлять камерой видеоняню Samsung SNH-V6410PNW можно подключить к домашней сети Wi-Fi высокое качество связи и изображение в формате высокой четкости Full HD 1080p, обеспечивающие реалистичность получаемой картинки при необходимости качество изображения можно снизить выбор зоны отслеживания (от 1 до 3) камера активируется при движении или плаче отправляя предупреждение на подключенное устройство видеоняня SmartCam SNH-V6410PNW автоматически поворачивается следя за передвижением малыша наличие двухсторонней связи - можно удаленно разговаривать с ребенком в случае недостаточной видимости включается система ночного видения поддержка карт памяти формата SDXC объемом до 128 Гб камера позволяет снимать видео и фото, записывая их на карту памяти возможность удаленно отключить видеоняню SNH-V6410PNW для подключения камеры к выбранному устройству необходимо скачать бесплатное приложение в AppStore или GooglePlay в комплект входит крепление, позволяющее разместить видеоняню на стене модель отличается лаконичным дизайном, высоким качеством используемых при создании материалов и безопасностью ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ: скорость интернет-соединения: от 1,5 Мбит/сек сеть Wi-Fi: 802.11 b/g/n версия операционной системы для устройств Apple: iOS 6 или выше версия операционной системы для устройств на базе Android: не ниже 4.X. версия операционной системы для PC: Windows не ниже 7/8 или Mac 10.7 и выше ХАРАКТЕРИСТИКИ: возраст: с рождения материал: пластик страна производства: Китай'


for _ in text.split():
    print(_, '->', text_clf_svm.predict([_]) ) 

  

