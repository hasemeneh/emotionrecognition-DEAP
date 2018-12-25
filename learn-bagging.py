import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time

csv = pd.read_csv('data-bandpassed-pca.csv').values
print("done reading data")
csv = csv.T
data = csv[:96].T
label = csv[96]
csv = None
print("done adjusting data")
print("Data",data.shape)
print("label",label.shape)

# data = data.T[:32].T
start_time = time.time()
print("start learning")
model = BaggingClassifier(RandomForestClassifier(min_samples_split=20,n_jobs =-2,n_estimators =40))
# model.fit(data, label)
# print("scores to self", model.score(data,label))

scores = cross_val_score(model, data, label, cv=10)
end_time = time.time()
print "Random Forest", scores
print("done learning ",(end_time - start_time))
