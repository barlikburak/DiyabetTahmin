#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:52:27 2021

@author: burak
"""

import veriler
import gorsel
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

neighbors = np.arange(1,13)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for n, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k, metric='minkowski')   
    knn.fit(veriler.X_train, veriler.y_train.ravel())      
    train_accuracy[n] = knn.score(veriler.X_train, veriler.y_train)      
    test_accuracy[n] = knn.score(veriler.X_test, veriler.y_test)

gorsel.plot_show('K-NN Degisen Komsu Sayisi', neighbors, test_accuracy, train_accuracy, 'Test Dogrulugu', 'Egitim Dogrulugu', 'Komsu Sayisi', 'Dogruluk')


#Yukardaki verileri inceleyerek n_neighbors=9 verirsek
knn = KNeighborsClassifier(n_neighbors=9, metric='minkowski')   
knn.fit(veriler.X_train, veriler.y_train.ravel())
accuracy = knn.score(veriler.X_test, veriler.y_test)
gorsel.plot_confusion_matrix_show(knn, veriler.X_test, veriler.y_test, str(accuracy))