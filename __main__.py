import pandas as pd
import seaborn as sn
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from MultiClassifier import MultiClassifier

def load_mnist_train_test():
    # The digits dataset
    digits = datasets.load_digits()
    
    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    
    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test




X_train, X_test, y_train, y_test = load_mnist_train_test()

# Create a clf: a support vector clf
clf1 = svm.SVC(gamma=0.001)
clf2 = RandomForestClassifier()
clf3 = KNeighborsClassifier()
clf4 = GaussianNB()

classifier = MultiClassifier([clf1, clf2, clf3, clf4])



# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
predicted = classifier.predict(X_test)

data = {'y_Actual':    y_test,
        'y_Predicted': predicted
        }
df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
sn.heatmap(confusion_matrix, annot=True)
