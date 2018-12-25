from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D,MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Bidirectional
from keras.utils import np_utils
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import time
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
csv = pd.read_csv('drive/My Drive/skripsoy/dataset/data-bandpassed-pca-persec.csv').values
print("done reading data")
csv = csv.T
n_features = 64
n_label = 4
output_dims = 2
data = csv[:n_features].T
label = csv[n_features]
populated3dData = np.array([data[:63]])
populatedLabel = np.array([label[0]]) 
populatedTestScore = np.array([])
populatedTestAccuracy = np.array([])
for x in range(1,1280):
	populated3dData = np.append(populated3dData, np.array([data[(x*63):((x+1)*63)]]), axis=0) #np.append( , np.atleast_3d(), axis=2)
	populatedLabel = np.append(populatedLabel,np.array(label[(x*63)]))
kf = KFold(n_splits=10)
kf.get_n_splits(data)
print("done adjusting data")
# print(kf)  
i = 0
for train_index, test_index in kf.split(populated3dData):
	# print("TRAIN:", train_index, "TEST:", test_index)
	i=i+1
	X_train, X_test = populated3dData[train_index], populated3dData[test_index]
	y_train, y_test = populatedLabel[train_index], populatedLabel[test_index]
	bs=500 #jangan gede2 berat 5000 ga kuat
	nb_ep=200
	y_train = np_utils.to_categorical(y_train, output_dims)
	y_test = np_utils.to_categorical(y_test, output_dims)

	model = Sequential()
	model.add(LSTM(64, return_sequences=True,input_shape=(63, n_features)))  # returns a sequence of vectors of dimension 32
	model.add(Dropout(0.2)) 
	model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
	model.add(Dropout(0.2)) 
	model.add(LSTM(16, return_sequences=False))  # return a single vector of dimension 32
	#perhatikan untuk return_sequences sebelum dense layer harus false,sesama lstm true
	model.add(Dense(output_dims, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


	wt0_train=time.time()
	model.fit(X_train, y_train, batch_size=bs, epochs=nb_ep,  validation_data=(X_test, y_test))
	score = model.evaluate(X_test, y_test, verbose=0)
	print('Hasil lstm untuk data-', i)
	print('Test score:', score[0])
	populatedTestScore = np.append(populatedTestScore,score[0])
	print('Test accuracy:', score[1])
	populatedTestAccuracy = np.append(populatedTestAccuracy,score[1])
	model = None
	X_train, X_test = None, None
	y_train, y_test = None, None


print("score ",populatedTestScore)
print("average score" ,  np.average(populatedTestScore))
print("max score" , np.amax(populatedTestScore))
print("min score" , np.amin(populatedTestScore))

print("accuracy ",populatedTestAccuracy)
print("average accuracy" ,  np.average(populatedTestAccuracy))
print("max accuracy" , np.amax(populatedTestAccuracy))
print("min accuracy" , np.amin(populatedTestAccuracy))