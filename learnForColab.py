import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time

file_path = 'drive/My Drive/skripsoy/dataset/data-bandpassed-stft-pca.csv'
csv = pd.read_csv(file_path).values
print(file_path)
print("done reading data")
csv = csv.T
n_features = 64
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
model = RandomForestClassifier(min_samples_split=1000,n_jobs =-1,n_estimators =40)
model2 = ExtraTreesClassifier(min_samples_split=1000,n_jobs =-1,n_estimators =40)
# model.fit(data, label)
# print("scores to self", model.score(data,label))

scores = cross_val_score(model, data, label[0], cv=10)
end_time = time.time()
print( "Random Forest Arousal", scores)
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print("average" ,  np.average(scores))
print("max" , np.amax(scores))
print("min" , np.amin(scores))

scores = cross_val_score(model, data, label[1], cv=10)
end_time = time.time()
print( "Random Forest Valence", scores)
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print("average" ,  np.average(scores))
print("max" , np.amax(scores))
print("min" , np.amin(scores))

scores = cross_val_score(model, data, label[2], cv=10)
end_time = time.time()
print( "Random Forest Dominance", scores)
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print("average" ,  np.average(scores))
print("max" , np.amax(scores))
print("min" , np.amin(scores))


scores = cross_val_score(model, data, label[3], cv=10)
end_time = time.time()
print( "Random Forest Liking", scores)
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print("average" ,  np.average(scores))
print("max" , np.amax(scores))
print("min" , np.amin(scores))


#/////////////////////////////

scores = cross_val_score(model2, data, label[0], cv=10)
end_time = time.time()
print("extrem Random Forest Arousal", scores)
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print("average" ,  np.average(scores))
print("max" , np.amax(scores))
print("min" , np.amin(scores))

scores = cross_val_score(model2, data, label[1], cv=10)
end_time = time.time()
print("extrem Random Forest Valence", scores)
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print("average" ,  np.average(scores))
print("max" , np.amax(scores))
print("min" , np.amin(scores))

scores = cross_val_score(model2, data, label[2], cv=10)
end_time = time.time()
print("extrem Random Forest Dominance", scores)
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print("average" ,  np.average(scores))
print("max" , np.amax(scores))
print("min" , np.amin(scores))


scores = cross_val_score(model2, data, label[3], cv=10)
end_time = time.time()
print("extrem Random Forest Liking", scores)
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print("average" ,  np.average(scores))
print("max" , np.amax(scores))
print("min" , np.amin(scores))

