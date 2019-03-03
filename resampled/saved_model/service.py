from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier , RadiusNeighborsClassifier
from sklearn.metrics import confusion_matrix

valenceModel = None
arousalModel = None
class SimpleChat(WebSocket):

    def handleMessage(self):
       #self adalah client yang mengirim data
       # print(self.data)
       data = np.array(json.loads(self.data)).astype(float)
       data = data[:n_features]
       print(data.shape)
       global valenceModel
       global arousalModel
       predictedValence = valenceModel.predict([data])
       predictedArousal = arousalModel.predict([data])
       print(predictedArousal)
       data = np.array([])
       data = np.append(data,predictedValence)
       data = np.append(data,predictedArousal)
       print(data.shape)
       data = json.dumps(data.tolist())
       print(data)
       self.sendMessage((u""+data))
       # for client in clients:
       #    if client != self:
       #       client.sendMessage(self.address[0] + u' - ' + self.data)

    def handleConnected(self):
       #self adalah client yang terkoneksi
       print(self.address, 'connected')

    def handleClose(self):
       #self adalah client yang terputus
       print(self.address, 'closed')

if __name__ == '__main__':
	
	n_features = 128
	n_label = 1

	valenceModel = pickle.load(open('valence/valence.model', 'rb'))
	arousalModel = pickle.load(open('arousal/arousal.model', 'rb'))

	csvArousal = pd.read_csv('arousal/test_set.csv').values
	csvValence = pd.read_csv('valence/test_set.csv').values
	csvArousal = csvArousal.T
	csvValence = csvValence.T
	dataArousal = csvArousal[:n_features].T
	labelArousal = csvArousal[n_features:(n_features+n_label)]
	dataValence = csvValence[:n_features].T
	labelValence = csvValence[n_features:(n_features+n_label)]
	scoreValence = valenceModel.score(dataValence,labelValence.ravel())
	scoreArousal = arousalModel.score(dataArousal,labelArousal.ravel())
	predictedValence = valenceModel.predict(dataValence)
	predictedArousal = arousalModel.predict(dataArousal)
	confuscious_matrixValence = confusion_matrix(labelValence.ravel(), predictedValence)
	confuscious_matrixArousal = confusion_matrix(labelArousal.ravel(), predictedArousal)
	print("Accuracy of valence Model :",scoreValence)
	print("Accuracy of arousal Model :",scoreArousal)
	clients = []
	print("confusion matrix valence")
	print(confuscious_matrixValence)
	print("confusion matrix arousal")
	print(confuscious_matrixArousal)
	server = SimpleWebSocketServer('127.0.0.1', 8000, SimpleChat)
	server.serveforever()

