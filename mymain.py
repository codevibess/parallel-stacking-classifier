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
from mpi4py import MPI
import timeit
import StackingClassifier as st


def bcast_data(data):
    print(f'[INFO] Bcasting data from the root process ({rank})') if rank == 0 else None
    bcast_start_time = MPI.Wtime()
    X_train, X_test, y_train, y_test = comm.bcast(data, root=0)
    bcast_finish_time = MPI.Wtime()

    bcast_time = bcast_finish_time - bcast_start_time
    print(f'[TIME] Master process ({rank}) finished Bcasting data with time {bcast_time}') if rank == 0 else print(f'[TIME] Process {rank} finished receive bcasted data with time {bcast_time}')
    return X_train, X_test, y_train, y_test 

def classify(X_train, X_test, y_train, y_test):
        # classification 
    algorithm=None
    classification_time_start = MPI.Wtime()
    if rank == 0:
        algorithm = 'knn'
        clf0 = KNeighborsClassifier()
        st.fit(clf0, X_train, y_train)
        classification_output = st.predict(clf0, X_test)
        pass
    elif rank == 1:
        algorithm='svc'
        clf1 = svm.SVC(gamma=0.001)
        st.fit(clf1, X_train, y_train)
        classification_output = st.predict(clf1, X_test)
        pass
    elif rank == 2:
        algorithm='gaussian'
        clf2 = GaussianNB()
        st.fit(clf2, X_train, y_train)
        classification_output = st.predict(clf2, X_test)
        pass
    elif rank == 3:
        algorithm='randomforest'
        clf3 = RandomForestClassifier()
        st.fit(clf3, X_train, y_train)
        classification_output = st.predict(clf3, X_test)

        pass

    classification_time_end = MPI.Wtime()
    classification_time = classification_time_end - classification_time_start
    print(f'[TIME] Process {rank} finished classification by {algorithm} algorithm with time: {classification_time}')
    return classification_output



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

def load_letter_data():
    data = pd.read_csv("letter-recognition/letter-recognition.data", header=None)
    y = np.array(data.iloc[:,0])  
    X = np.array(data.iloc[:,1:len(data)])  
    return X, y 


def train_test( X, y ):
    if rank == 0:
        program_start_time = MPI.Wtime()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False)

        data = (X_train, X_test, y_train, y_test)
    else:
        data = None
        program_start_time = MPI.Wtime()
        

    X_train, X_test, y_train, y_test = bcast_data(data)

    classification_output = classify(X_train, X_test, y_train, y_test)
    outputs_from_classifications = comm.gather(classification_output)
    # stacking
    if rank == 0:
        voted_data = st.vote(outputs_from_classifications)
        acc = accuracy_score(voted_data, y_test)
        print(acc)
    program_finish_time = MPI.Wtime()
    print(program_finish_time-program_start_time, rank)


def cross_validation(X, y):
    if rank == 0:
        kf = KFold(n_splits=10, shuffle = True)
        kfold_array = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            kfold_array.append((X_train, X_test, y_train, y_test))
        data = (kfold_array)
    else:
        data = None


    data = comm.bcast(data)
    accuracies = list()
    count = 0 
    for tuple_with_data in data: 
        count += 1
        print(f"[INFO] Running cross_validation with {count} chunk of data by {rank} process")
        X_train, X_test, y_train, y_test = tuple_with_data
        classification_output = classify(X_train, X_test, y_train, y_test)
        outputs_from_classifications = comm.gather(classification_output)
        # stacking
        if rank == 0:
            voted_data = st.vote(outputs_from_classifications)
            acc = accuracy_score(voted_data, y_test)
            
            accuracies.append(acc)
        comm.barrier()
    
    if rank == 0:
        acc_final = np.mean(accuracies)
        print(f'[ACCURANCY] Final accurancy with CV chunks is {acc_final}')
    



# initialize MPI environment
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


if rank==0:
    print(f"[INFO] Program runned in {size} processes")

print(f"[INFO] Hello from process number {rank}")
X, y = load_mnist_data()
program_start_time = MPI.Wtime()
# classification_output = train_test(X, y)

classification_output = cross_validation(X, y)
program_end_time = MPI.Wtime()
program_time = program_end_time - program_start_time


if rank == 0:
    print(f'[INFO] Stacking classifier finish work with time: {program_time}')
# MPI environment finalization
MPI.Finalize()





