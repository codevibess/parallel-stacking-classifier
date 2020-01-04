import sys
import pandas as pd
import seaborn as sn
import time
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from MultiClassifier import MultiClassifier
from sklearn.model_selection import KFold

import timeit
import dataset_loader as loader

num_cores = 1
num_repetitions = 10


def print_results(description, accuracy, time, num_cores):
    print(description)
    print("Accuracy: " + str(accuracy), ", Time: " +
          str(time) + " seconds" + ", cores: " + str(num_cores))


def get_multi_classifier():
    clf1 = RidgeClassifier()
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = LinearDiscriminantAnalysis()
    clf4 = GaussianNB()
    classifier = MultiClassifier([
        clf1,
        clf2,
        clf3,
        clf4
    ])
    return classifier


def train_test_run(X, y, num_cores, description):
    print("Train-test run of " + description)
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
    print("CV run of " + description)
    start = time.time()
    kf = KFold(n_splits=10, shuffle=True)
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


if sys.argv[1] == 'MNIST':
    X, y = loader.load_mnist_data()
elif sys.argv[1] == 'CIFAR-10':
    X, y = loader.load_cifar10_data()
elif sys.argv[1] == 'CIFAR-100':
    X, y = loader.load_cifar100_data()
elif sys.argv[1] == 'letter-recognition':
    X, y = loader.load_letter_data()


if sys.argv[2] == 'CV':
    classification_output = cross_validation_run(X, y, num_cores, f'{sys.argv[1]} {sys.argv[2]}')
elif sys.argv[2] == 'test-train':
    classification_output = train_test_run(X, y, num_cores, f'{sys.argv[1]} {sys.argv[2]}')

