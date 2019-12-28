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

def load_cifar10_data():
    data = list()
    labels = list()
    for batch_index in range(1, 6):
        batch = unpickle("cifar-10-batches-py/data_batch_" + str(batch_index)) 
        data = data + list(batch[b"data"])
        labels = labels + batch[b"labels"]
    return data, labels
    
def load_cifar100_data():
    data = list()
    labels = list()
    for batch_name in ["test", "train"]:
        batch = unpickle("cifar-100/" + batch_name) 
        data = data + list(batch[b"data"])
        labels = labels + batch[b"coarse_labels"]
    return data, labels

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_letter_data()
    data = pd.read_csv("letter-recognition/letter-recognition.data", header=None)
    y = np.array(data.iloc[:,0])  
    X = np.array(data.iloc[:,1:len(data)])  
    return X, y 

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
        
X_mnist, y_mnist = load_mnist_data()
X_cifar10, y_cifar10 = load_cifar10_data()
X_cifar100, y_cifar100 = load_cifar100_data()
X_letter, y_letter = load_letter_data()

train_test_run(X_mnist, y_mnist, num_cores, "Train-test MNIST")
cross_validation_run(X_mnist, y_mnist, num_cores, "CV MNIST")

train_test_run(X_cifar10, y_cifar10, num_cores, "Train-test CIFAR-10")
cross_validation_run(X_cifar10, y_cifar10, num_cores, "CV CIFAR-10")

train_test_run(X_cifar10, y_cifar10, num_cores, "Train-test CIFAR-100")
cross_validation_run(X_cifar10, y_cifar10, num_cores, "CV CIFAR-100")

train_test_run(X_letter, y_letter, num_cores, "Train-test letter recognition")
cross_validation_run(X_letter, y_letter, num_cores, "CV letter recognition")
