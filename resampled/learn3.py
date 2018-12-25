import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier , AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier , RadiusNeighborsClassifier

import numpy as np
from numpy import linalg as LA

def cosine(X, Y):
	# X=np.array(X.todense())
	# Y=np.array(Y.todense())
	norm = LA.norm(X) * LA.norm(Y)
	return np.dot(X, Y.T)/norm

csv = pd.read_csv('data-bandpassed-theta-alpha-beta-gamma-pca-63DetikAllchannel-rounded-smote-Arousal.csv').values
print("done reading data")
csv = csv.T
n_features = 128
n_label = 1
data = csv[:n_features].T
label = csv[n_features:(n_features+n_label)]
# label = label[0]
label = label.T
print(label)
arr = np.arange(data.shape[0])
np.random.shuffle(arr)
data = data[arr]
label = label[arr]
label = label.T
if __name__ == '__main__':
	# for x in xrange(1,100):	
	start_time = time.time()
	print("start learning")
	# model = (RandomForestClassifier(criterion='gini',min_samples_leaf =1,min_samples_split=(2*x),n_jobs =-1,bootstrap =True,n_estimators =10,verbose=0))
	model = BaggingClassifier(base_estimator=KNeighborsClassifier(weights='uniform',p=2,n_neighbors=1,leaf_size=128,algorithm='kd_tree'),max_samples =0.8,n_estimators=200,n_jobs=-1,bootstrap_features=True,warm_start=False)
	# model = KNeighborsClassifier(weights='uniform',p=2,n_neighbors=1,leaf_size=128,algorithm='auto',n_jobs=-1)
	# model = RadiusNeighborsClassifier(weights='uniform',p=2,radius=10000.0,leaf_size=128,algorithm='auto',n_jobs=-1)
	kf = KFold(n_splits=10)
	scores = []
	conf_matrix = np.array([[0,0],[0,0]])
	label= label.ravel()
	for train_index, test_index in kf.split(data):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = data[train_index], data[test_index]
		y_train, y_test = label[train_index], label[test_index]
		model.fit(X_train,y_train)
		y_pred = model.predict(X_test)
		score = model.score(X_test,y_test)
		scores = np.append(scores,score)
		confuscious_matrix = confusion_matrix(y_test, y_pred)
		conf_matrix = conf_matrix + confuscious_matrix

	
	end_time = time.time()
	print "Random Forest Arousal", scores
	print("done learning ",(end_time - start_time))
	scores = np.array(scores)
	print "average" ,  np.average(scores)
	print "max" , np.amax(scores)
	print "min" , np.amin(scores)
	print(conf_matrix)
