import pandas as pd
import seaborn as sn
import time
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from MultiClassifier import MultiClassifier
import timeit

num_cores = 1
num_repetitions = 10

def load_mnist_train_test():
    # The digits dataset
    digits = datasets.load_digits()
    
    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    
    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.3, shuffle=False)
    return X_train, X_test, y_train, y_test

def print_results(description, accuracy, time, num_cores):
    print(description)
    print("Accuracy: " + str(accuracy), ", Time: " + str(time) + " seconds" + ", cores: " + str(num_cores))

def train_test_run(X_train, X_test, y_train, y_test, num_cores):
    start = time.time()
    clf1 = svm.SVC(gamma=0.001)
    clf2 = RandomForestClassifier()
    clf3 = KNeighborsClassifier()
    clf4 = GaussianNB()
    classifier = MultiClassifier([clf1, clf2, clf3, clf4])
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    acc = accuracy_score(predicted, y_test)
    end = time.time()
    print_results("Train-test MNIST", acc, end - start, num_cores)

X_train, X_test, y_train, y_test = load_mnist_train_test()
train_test_run(X_train, X_test, y_train, y_test, num_cores)

