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
    print("Accuracy: " + str(accuracy), ", Time: " + str(time) + " seconds" + ", cores: " + str(num_cores))

def get_multi_classifier():
    clf1 = RidgeClassifier()
    clf2 = RandomForestClassifier(n_estimators = 10)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
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

X_mnist, y_mnist = loader.load_mnist_data()
X_cifar10, y_cifar10 = loader.load_cifar10_data()
X_cifar100, y_cifar100 = loader.load_cifar100_data()
X_letter, y_letter = loader.load_letter_data()
print("data loaded")
train_test_run(X_mnist, y_mnist, num_cores, "Train-test MNIST")
cross_validation_run(X_mnist, y_mnist, num_cores, "CV MNIST")

train_test_run(X_cifar10, y_cifar10, num_cores, "Train-test CIFAR-10")
cross_validation_run(X_cifar10, y_cifar10, num_cores, "CV CIFAR-10")

train_test_run(X_cifar100, y_cifar100, num_cores, "Train-test CIFAR-100")
cross_validation_run(X_cifar100, y_cifar100, num_cores, "CV CIFAR-100")

train_test_run(X_letter, y_letter, num_cores, "Train-test letter recognition")
cross_validation_run(X_letter, y_letter, num_cores, "CV letter recognition")
