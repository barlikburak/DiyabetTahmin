#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:52:27 2021

@author: burak
"""

#1 Kutuphaneler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#2 Veri Yukleme
veriler = pd.read_csv('veri_seti.csv')

#3 Eksik Verilerin Doldurulmasi

#3.1 Degeri 0 olanlari bos olarak belirttik.
veriler[['Glikoz','Tansiyon','Cilt Kalinligi','Insulin',
         'Vucut Kitle Indexi']] = veriler[['Glikoz','Tansiyon'
                                           ,'Cilt Kalinligi','Insulin',
                                           'Vucut Kitle Indexi']].replace(0,
                                                                          np.NaN)
#3.2 Bos olan degerlerin yerine sutundaki degerlerin ortalamasını yerlesmesini saglayan model olusturduk
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#3.3 Ogretiyor
imputer.fit(veriler.iloc[:,1:6].values)
#3.4 Uyguluyor, Donusturuyor
veriler.iloc[:,1:6] = imputer.transform(veriler.iloc[:,1:6].values)

#4 Verinin Bölünmesi

#4.1 DataFrame Dilimleme (slice)
x = veriler.iloc[:,0:8]
y = veriler.iloc[:,8:]

#4.2 Numpy Dizi(array) Donusumu
X = x.values
Y = y.values

#4.3 Verilerin Egitim Ve Test Icin Bolunmesi
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#5 Verilerin Olceklenmesi
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
