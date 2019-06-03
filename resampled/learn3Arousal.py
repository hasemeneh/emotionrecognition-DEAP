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
import pickle
import numpy as np
from numpy import linalg as LA


csv = pd.read_csv('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded-smote-arousal.csv').values
print("done reading data")
csv = csv.T
n_features = 128
n_label = 1
data = csv[:n_features].T
label = csv[n_features:(n_features+n_label)]
max_acc = 0
label = label.T
print(label)
arr = np.arange(data.shape[0])
np.random.shuffle(arr)
data = data[arr]
label = label[arr]
label = label.T
if __name__ == '__main__':
	start_time = time.time()
	print("start learning")
	model = KNeighborsClassifier(weights='uniform',p=2,n_neighbors=1,leaf_size=7,algorithm='kd_tree',n_jobs=-1)
	kf = KFold(n_splits=10)
	scores = []
	save_model = None
	conf_matrix = np.array([[0,0],[0,0]])
	label= label.ravel()
	for train_index, test_index in kf.split(data):
		#melakukan K folding dan cross validation 
		X_train, X_test = data[train_index], data[test_index]
		y_train, y_test = label[train_index], label[test_index]
		model.fit(X_train,y_train)
		y_pred = model.predict(X_test)
		score = model.score(X_test,y_test)
		print(score)
		scores = np.append(scores,score)
		if max_acc < score:
			#jika akurasi maximal lebih besar dari training sebelumnya maka akan di simpan
			save_model = model
			max_acc = score
			test_set = np.vstack(([np.zeros(n_features+n_label)],np.hstack((X_test,np.array([y_test]).T))))
			train_set = np.vstack(([np.zeros(n_features+n_label)],np.hstack((X_train,np.array([y_train]).T))))
			filename = 'arousal.model'
			pickle.dump(model,open(filename, 'wb'))
			np.savetxt('train_set.csv', train_set, delimiter=',')
			np.savetxt('test_set.csv', test_set, delimiter=',')
		confuscious_matrix = confusion_matrix(y_test, y_pred)
		conf_matrix = conf_matrix + confuscious_matrix

	
	end_time = time.time()
	print "Arousal", scores
	print("done learning ",(end_time - start_time))
	scores = np.array(scores)
	print "average" ,  np.average(scores)
	print "max" , np.amax(scores)
	print "min" , np.amin(scores)
	print(conf_matrix)
