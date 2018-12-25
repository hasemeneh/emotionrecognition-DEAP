import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter 
csv = pd.read_csv('data-bandpassed-alpha-beta-gamma-pca3Value.csv').values
print("done reading data")
csv = csv.T
n_features = 128
n_label = 3
data = csv[:n_features].T
label = csv[n_features:(n_features+n_label)]
label = label.T
arr = np.arange(1280)
np.random.shuffle(arr)
data = data[arr]
label = label[arr]
label = label.T
csv = None
print("done adjusting data")
print("Data",data.shape)
print("label",label.shape)

# data = data.T[:32].T
averageOfArousal = np.array([])
averageOfValence = np.array([])
averageOfDominance = np.array([])
averageOfLiking = np.array([])
extremeAverageOfArousal = np.array([])
extremeAverageOfValence = np.array([])
extremeAverageOfDominance = np.array([])
extremeAverageOfLiking = np.array([])
y = xrange(5,20)
# for x in y:
start_time = time.time()
print("start learning")
model = RandomForestClassifier(min_samples_leaf =1,min_samples_split=2,n_jobs =-2,n_estimators =20,verbose=1)
model2 = ExtraTreesClassifier(min_samples_leaf =1,min_samples_split=2,n_jobs =-2,n_estimators =20,verbose=1)
# model.fit(data, label)
# print("scores to self", model.score(data,label))

scores = cross_val_score(model, data, label[0], cv=10)
end_time = time.time()
print "Random Forest Arousal", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
averageOfArousal = np.append(averageOfArousal,np.average(scores))
print "max" , np.amax(scores)
print "min" , np.amin(scores)

scores = cross_val_score(model, data, label[1], cv=10)
end_time = time.time()
print "Random Forest Valence", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
averageOfValence = np.append(averageOfValence,np.average(scores))
print "max" , np.amax(scores)
print "min" , np.amin(scores)

scores = cross_val_score(model, data, label[2], cv=10)
end_time = time.time()
print "Random Forest 4 values", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
averageOfDominance = np.append(averageOfDominance,np.average(scores))
print "max" , np.amax(scores)
print "min" , np.amin(scores)



#/////////////////////////////

scores = cross_val_score(model2, data, label[0], cv=10)
end_time = time.time()
print "extrem Random Forest Arousal", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
extremeAverageOfArousal = np.append(extremeAverageOfArousal,np.average(scores))
print "max" , np.amax(scores)
print "min" , np.amin(scores)

scores = cross_val_score(model2, data, label[1], cv=10)
end_time = time.time()
print "extrem Random Forest Valence", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
extremeAverageOfValence = np.append(extremeAverageOfValence,np.average(scores))
print "max" , np.amax(scores)
print "min" , np.amin(scores)

scores = cross_val_score(model2, data, label[2], cv=10)
end_time = time.time()
print "extrem Random Forest 4 values", scores
print("done learning ",(end_time - start_time))
scores = np.array(scores)
print "average" ,  np.average(scores)
extremeAverageOfDominance = np.append(extremeAverageOfDominance,np.average(scores))
print "max" , np.amax(scores)
print "min" , np.amin(scores)


# plt.subplot(421)
# plt.yscale('linear')
# # plt.axis([0, 10, 0, 1])
# plt.plot(y,averageOfArousal)
# plt.title('averageOfArousal')
# # plt.grid(True)
# plt.subplot(422)
# plt.yscale('linear')
# # plt.axis([0, 10, 0, 1])
# plt.plot(y,averageOfValence)
# plt.title('averageOfValence')
# # plt.grid(True)
# plt.subplot(423)
# plt.yscale('linear')
# # plt.axis([0, 10, 0, 1])
# plt.plot(y,averageOfDominance)
# plt.title('averageOfDominance')
# # plt.grid(True)
# plt.subplot(424)
# plt.yscale('linear')
# # plt.axis([0, 10, 0, 1])
# plt.plot(y,averageOfLiking)
# plt.title('averageOfLiking')
# # plt.grid(True)
# plt.subplot(425)
# plt.yscale('linear')
# # plt.axis([0, 10, 0, 1])
# plt.plot(y,extremeAverageOfArousal)
# plt.title('extremeAverageOfArousal')
# # plt.grid(True)
# plt.subplot(426)
# plt.yscale('linear')
# # plt.axis([0, 10, 0, 1])
# plt.plot(y,extremeAverageOfValence)
# plt.title('extremeAverageOfValence')
# # plt.grid(True)
# plt.subplot(427)
# plt.yscale('linear')
# # plt.axis([0, 10, 0, 1])
# plt.plot(y,extremeAverageOfDominance)
# plt.title('extremeAverageOfDominance')
# # plt.grid(True)
# plt.subplot(428)
# plt.yscale('linear')
# # plt.axis([0, 10, 0, 1])
# plt.plot(y,extremeAverageOfLiking)
# plt.title('extremeAverageOfLiking')
# # plt.grid(True)

# plt.gca().yaxis.set_minor_formatter(NullFormatter())
# # Adjust the subplot layout, because the logit one may take more space
# # than usual, due to y-tick labels like "1 - 10^{-3}"
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
# plt.show()

