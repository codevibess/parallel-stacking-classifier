import pandas as pd
import seaborn as sn
import time
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from MultiClassifier import MultiClassifier
from sklearn.model_selection import KFold

import timeit

num_cores = 1
num_repetitions = 10

def load_mnist_data():
    # The digits dataset
    digits = datasets.load_digits()
    
    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target

def print_results(description, accuracy, time, num_cores):
    print(description)
    print("Accuracy: " + str(accuracy), ", Time: " + str(time) + " seconds" + ", cores: " + str(num_cores))

def get_multi_classifier():
    clf1 = svm.SVC(gamma=0.001)
    clf2 = RandomForestClassifier()
    clf3 = KNeighborsClassifier()
    clf4 = GaussianNB()
    classifier = MultiClassifier([clf1, clf2, clf3, clf4])
    return classifier

def train_test_run(X, y, num_cores, description):
    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False)
    start = time.time()
    classifier = get_multi_classifier()
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    acc = accuracy_score(predicted, y_test)
    end = time.time()
    print_results(description, acc, end - start, num_cores)

def cross_validation_run(X, y, num_cores, description):
    start = time.time()
    kf = KFold(n_splits=10, shuffle = True)
    kf.get_n_splits(X)
    accuracies = list()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = get_multi_classifier()
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_test)
        acc = accuracy_score(predicted, y_test)
        accuracies.append(acc)
    acc = np.mean(accuracies)
    end = time.time()
    print_results(description, acc, end - start, num_cores)    
        
X, y = load_mnist_data()
train_test_run(X, y, num_cores, "Train-test MNIST")
cross_validation_run(X, y, num_cores, "CV MNIST")
