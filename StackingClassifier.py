import numpy as np


def fit(clf, X_train, y_train):
    return clf.fit(X_train, y_train)


def predict(clf, X_test):
    return clf.predict(X_test)


def vote(preds):
    print('__________', preds)
    np_preds = np.array(preds).T
    result_list = list()
    for row in np_preds:
        result_list.append(np.argmax(np.bincount(row)))
    return np.array(result_list)
