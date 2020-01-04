import numpy as np


class MultiClassifier:
    classifiers = list()

    def __init__(self, classifier_list):
        self.classifiers = classifier_list

    def fit(self, X_train, y_train):
        for clf in self.classifiers:
            clf.fit(X_train, y_train)

    def predict(self, X_test):
        preds = list()
        for clf in self.classifiers:
            preds.append(clf.predict(X_test))
        aggregated_preds = self.vote(preds)
        return aggregated_preds

    def vote(self, preds):
        np_preds = np.array(preds).T
        result_list = list()
        for row in np_preds:
            result_list.append(np.argmax(np.bincount(row)))
        return np.array(result_list)
