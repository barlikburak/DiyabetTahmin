#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:52:27 2021

@author: burak
"""

import veriler
import gorsel
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(veriler.X_train, veriler.y_train.ravel())
accuracy = gnb.score(veriler.X_test, veriler.y_test)
gorsel.plot_confusion_matrix_show(gnb, veriler.X_test, veriler.y_test, str(accuracy))