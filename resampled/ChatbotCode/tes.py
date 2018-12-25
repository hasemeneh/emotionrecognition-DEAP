from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

from svm import Cossim

X = np.array([[1,1], [0,0], [1,0], [0,1]])
y = np.array([1, 1, 0, 0])


for clf, name in [(SVC(kernel=Cossim, C=1000), 'pykernel')]:
    clf.fit(X, y)
    print (name)
    print (clf)
    print ('Predictions:', clf.predict(X))
    print ('Accuracy:', accuracy_score(clf.predict(X), y))