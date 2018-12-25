import numpy as np
import math
from numpy import linalg as LA
from sklearn.preprocessing import normalize
from scipy import sparse
# def Cossim(X, Y):
#     norm_1 = np.sqrt((X ** 2).sum(axis=1)).reshape(X.shape[0], 1)
#     norm_2 = np.sqrt((Y ** 2).sum(axis=1)).reshape(Y.shape[0], 1)
#     return X.dot(Y.T) / (norm_1 * norm_2.T)

def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according 
    to the definition of the dot product
    """
    dot_product = np.dot(a.reshape(-1,1), b.reshape(-1,1).T)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def map_func(x):
    if len(x.shape) == 1:
        return np.r_[x, calc_z(x)]
    else:
        z = [np.r_[j, calc_z(j)] for j in x]
        return np.array(z)

def calc_z(x):
    return np.dot(x, x.T)

def my_kernel(X, Y):
    # X=np.array(X.todense())
    # Y=np.array(Y.todense())
    # print("titid",X.ravel().shape,Y.shape)
    # print(np.array(X.todense()))
    # print(Y)
    # print(type())
    # norm = LA.norm(X) * LA.norm(Y)
    # # print("titid",X.shape,Y.shape)
    # # norm = math.sqrt(np.dot(X,X)) * math.sqrt(np.dot(Y,Y))
    # # norm = normalizer(X) * normalizer(Y)
    # # print(LA.norm(X))
    # # print(np.dot(X.ravel(), Y.ravel()))
    # # return sparse.csr_matrix(np.dot(X, Y.T)/norm)
    # return (np.dot(X, Y.T)/norm)
    kernel = np.ones((data_1.shape[0], data_2.shape[0]))
    for d in range(data_1.shape[1]):
        column_1 = data_1[:, d].reshape(-1, 1)
        column_2 = data_2[:, d].reshape(-1, 1)
        if self._c is None:
            kernel *= self._h( (column_1 - column_2.T) / self._a )
        else:
            kernel *= self._h( (column_1 - self._c) / self._a ) * self._h( (column_2.T - self._c) / self._a )

    return kernel
# def my_kernel(X, Y):
#     return np.dot(map_func(X), map_func(Y.T))

# def Cossim(X, Y):
#     def _compute(self, X, Y):
#         self._dim = X.shape[1]
#         norm_1 = np.sqrt((X ** 2).sum(axis=1)).reshape(X.shape[0], 1)
#         norm_2 = np.sqrt((Y ** 2).sum(axis=1)).reshape(Y.shape[0], 1)
#         return X.dot(Y.T) / (norm_1 * norm_2.T)

#     def dim(self):
#         return self._dim


# implementasi cosine
import copy
import time
from collections import Counter

########################### SPLIT DATASET ##########################
# split list
def splitList(arr, test_size):
  a = arr[int(len(arr) * 0) : int(len(arr) * test_size)]
  b = arr[int(len(arr) * test_size) : int(len(arr) * 1)]
  return a, b

def test_train_split(x, y, test_size, randomize=False):
  if randomize == True :
      random.shuffle(x)
      random.shuffle(y)

  x_test, x_train = splitList(x, test_size)
  y_test, y_train = splitList(y, test_size)

  return x_test,x_train,y_test,y_train

# def train(X, Y, lRate = 0.0, tolerance = 0.0):
#     X_count = X.shape[0]
#     alpha = np.zeros(X_count)
#     # Gram matrix
#     K = np.zeros((X_count, X_count))
#     for i in range(X_count):
#         for j in range(X_count):
#             K[i,j] = cos_sim(X[i], X[j])

#     max_iterations = 1000
#     for ite in range(max_iterations):
#         for i in range(X.shape[0]):
#             jum = 0
#             val = 0
#             for j in range(X.shape[0]):
#                 val= alpha[j] * Y[j] * K[i,j] #rbf(X[i],X[j])
#                 jum = jum + val
#             if jum <= 0:
#                 val = -1
#             elif jum >0:
#                 val = 1
#             if val != Y[i]:
#                 alpha[i] = alpha[i] + 1
#     return alpha

def train(X, Y, lRate = 0.0, tolerance = 0.0):
    X_count = X.shape[0]
    alpha = np.zeros(X_count)
    K = cos_sim(X, Y)



    # for i in range(X_count):
    #     for j in range(X_count):
    #         K[i,j] = cos_sim(X[i], X[j])
    return K
    # max_iterations = 1000
    # for ite in range(max_iterations):
    #     for i in range(X.shape[0]):
    #         jum = 0
    #         val = 0
    #         for j in range(X.shape[0]):
    #             val= alpha[j] * Y[j] * K[i,j] #rbf(X[i],X[j])
    #             jum = jum + val
    #         if jum <= 0:
    #             val = -1
    #         elif jum >0:
    #             val = 1
    #         if val != Y[i]:
    #             alpha[i] = alpha[i] + 1
    # return alpha

def score(train_x,train_y,test_x,test_y,alpha):
    m = test_y.size
    right = 0

    for i in range(m):
        s = 0
        for a, train_x,train_y  in zip(alpha, train_x, train_y):
            s += a * train_y * cos_sim(test_x[i], train_x)
        if s >0:
            s = 1
        elif s <=0:
            s = -1
        if test_y[i] == s:
            right +=1

    print (" Correct : ",right," Accuracy : ",right*100/test_x.shape[0])
        #return y_predict

# a = np.array([[3,2,5,1,1,3,21,64,72,24,42],
#     [57,2,5,84,5,7,21,13,23,37,3],
#     [45,2,24,83,2,8,54,2,74,23,54],
#     [36,35,9,2,75,75,32,69,42,32,34],
#     [45,7,34,97,79,0,60,1,97,42,98],
#     [24,22,1,56,6,12,9,0,12,7,0],
#     [64,72,64,85,3,3,24,22,96,2,14]])

# b = np.array([34,8,12,0,1,9,2])

# train_x, test_x, train_y, test_y = test_train_split(a,b, test_size=0.3)
# print(len(train_x))
# for i in range(len(train_x)):
#     alpha = train(train_x[i], train_y, 0.0, 0.0)
#     print(alpha)
#     score(train_x, train_y, test_x, test_y, alpha)





# class SVM():
#     def __init___(self, kernel=cosine):
#         self.kernel = kernel

#     def fit(self, data, label):
#         self.data = data
#         self.label = label

#         transform()

#     def train(self, data, label):
#         classification = np.dot(np.array(data),np.array(label))/
#             (np.linalg.norm(data) * np.linalg.norm(label))
#         return classification