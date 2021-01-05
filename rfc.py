#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:52:27 2021

@author: burak
"""

import veriler
import gorsel
import numpy as np
from sklearn.ensemble import RandomForestClassifier

estimators = np.arange(1,100)
train_accuracy = np.empty(len(estimators))
test_accuracy = np.empty(len(estimators))

for n, k in enumerate(estimators):
    rfc = RandomForestClassifier(n_estimators=k, criterion = 'entropy') 
    rfc.fit(veriler.X_train, veriler.y_train.ravel())      
    train_accuracy[n] = rfc.score(veriler.X_train, veriler.y_train)      
    test_accuracy[n] = rfc.score(veriler.X_test, veriler.y_test)
   
gorsel.plot_show('Random Forest Degisen Agac Sayisi', estimators, test_accuracy, train_accuracy, 'Test Dogrulugu', 'Egitim Dogrulugu', 'Agac Sayisi', 'Dogruluk')

#Yukardaki verileri inceleyerek n_estimators=50 verirsek
rfc = RandomForestClassifier(n_estimators=50, criterion = 'entropy')
rfc.fit(veriler.X_train, veriler.y_train.ravel())
accuracy = rfc.score(veriler.X_test, veriler.y_test)
gorsel.plot_confusion_matrix_show(rfc, veriler.X_test, veriler.y_test, str(accuracy))