#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:52:27 2021

@author: burak
"""

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
    
def plot_show(title, arange, test_accuracy, train_accuracy, test_label, train_label, x_label, y_label):
    plt.title(title)
    plt.plot(arange, test_accuracy, label=test_label)
    plt.plot(arange, train_accuracy, label=train_label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
    
def plot_confusion_matrix_show(algorithm, X_test, y_test, accuracy):
    plot_confusion_matrix(algorithm, X_test, y_test)
    plt.title("Dogruluk=" + accuracy)
    plt.show()