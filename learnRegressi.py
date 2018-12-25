import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor , ExtraTreesRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time

csv = pd.read_csv('data-bandpassed-theta-alpha-beta-gamma-pca-forRegression.csv').values
print("done reading data")
csv = csv.T
n_features = 128
n_label = 4
data = csv[:n_features].T
label = csv[n_features:(n_features+n_label)]
csv = None
print("done adjusting data")
print("Data",data.shape)
print("label",label.shape)

# data = data.T[:32].T
start_time = time.time()
print("start learning")
model = RandomForestRegressor(min_samples_split=100,n_jobs =-2,n_estimators =10)
model2 = ExtraTreesRegressor(min_samples_split=100,n_jobs =-2,n_estimators =10)
# model.fit(data, label)
# print("scores to self", model.score(data,label))

scores = cross_val_score(model, data, label[0], cv=10)
end_time = time.time()
print "Random Forest Arousal", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
print "max" , np.amax(scores)
print "min" , np.amin(scores)

scores = cross_val_score(model, data, label[1], cv=10)
end_time = time.time()
print "Random Forest Valence", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
print "max" , np.amax(scores)
print "min" , np.amin(scores)

scores = cross_val_score(model, data, label[2], cv=10)
end_time = time.time()
print "Random Forest Dominance", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
print "max" , np.amax(scores)
print "min" , np.amin(scores)


scores = cross_val_score(model, data, label[3], cv=10)
end_time = time.time()
print "Random Forest Liking", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
print "max" , np.amax(scores)
print "min" , np.amin(scores)


#/////////////////////////////

scores = cross_val_score(model2, data, label[0], cv=10)
end_time = time.time()
print "extrem Random Forest Arousal", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
print "max" , np.amax(scores)
print "min" , np.amin(scores)

scores = cross_val_score(model2, data, label[1], cv=10)
end_time = time.time()
print "extrem Random Forest Valence", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
print "max" , np.amax(scores)
print "min" , np.amin(scores)

scores = cross_val_score(model2, data, label[2], cv=10)
end_time = time.time()
print "extrem Random Forest Dominance", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
print "max" , np.amax(scores)
print "min" , np.amin(scores)


scores = cross_val_score(model2, data, label[3], cv=10)
end_time = time.time()
print "extrem Random Forest Liking", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
print "max" , np.amax(scores)
print "min" , np.amin(scores)

