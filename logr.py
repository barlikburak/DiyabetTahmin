#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:52:27 2021

@author: burak
"""

import veriler
import gorsel
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(veriler.X_train, veriler.y_train.ravel())
accuracy = logr.score(veriler.X_test, veriler.y_test)
gorsel.plot_confusion_matrix_show(logr, veriler.X_test, veriler.y_test, str(accuracy))