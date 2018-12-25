from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.utils import shuffle
import scipy.sparse as sparse
import numpy as np


digits = load_digits(2)
X, y = shuffle(digits.data, digits.target)

gamma = 1.0


X_train, X_test = X[:100, :], X[100:, :]
y_train, y_test = y[:100], y[100:]

m1 = SVC(kernel='rbf',gamma=1)
print(m1.fit(X_train, y_train))
print(m1.predict(X_test))


def my_kernel(x,y):
    d = np.zeros((x.shape[0], y.shape[0]))
    for i, row_x in enumerate(x):
        for j, row_y in enumerate(y):
            d[i, j] = np.exp(-gamma * np.linalg.norm(row_x - row_y))

    return d

m2 = SVC(kernel=my_kernel)
print(m2.fit(X_train, y_train))
print(m2.predict(X_test))