from scipy.io import loadmat
import numpy as np
import preprocessing as prp
import thread
import time
import preprocessing as prp
from os import system, name 
from sklearn.decomposition import PCA
n_features = 128
n_label = 4
pca = PCA(n_components=n_features)
activeThread = 0
count = 0
populatedData = np.zeros((n_features+n_label))
def populatePart(n):
	data = np.zeros((n_features+n_label))
	label = np.zeros(n_label)
	arrayX = np.array
	vstackX = np.vstack
	hstackX = np.hstack
	appendX = np.append
	roundX = np.round
	zerosX = np.zeros
	multiplier = np.ones((60,1))
	matTemp = loadmat(str(n)+'.mat')
	labelTemp=matTemp['labels']
	dataTemp = matTemp['data']
	bandPassX =  prp.bandpassX
	global count
	for y in xrange(0,40):
		labelY = labelTemp[y]
		labelY = roundX(labelY)
		labelY = (labelY>=4.5).astype(int)
		dataY = dataTemp[y]
		dataY = dataY[:32].T
		# aplikasikan Bandpass filter
		bandPassed = bandPassX(dataY)
		dataY = None
		temporalData = bandPassed[0].T
		temporalData = vstackX((temporalData,bandPassed[1].T))
		temporalData = vstackX((temporalData,bandPassed[2].T))
		temporalData = vstackX((temporalData,bandPassed[3].T))
		bandPassed = None
		#reduksi dimensi dengan PCA
		tempPCA = pca.fit(temporalData.T).singular_values_
		tobeprocessed = arrayX(labelY)
		temporalData = hstackX((tempPCA,tobeprocessed))
		tempPCA = None
		tobeprocessed = None
		count = count+1
		data = vstackX((data,temporalData))
		temporalData = None
	global populatedData
	populatedData = vstackX((populatedData,data[1:]))
	global activeThread
	activeThread = activeThread -1
def clear(): 
  
    if name == 'nt': 
        _ = system('cls') 
  
    else: 
        _ = system('clear')

if __name__ == '__main__':
	i=1
	timecount = 0
	while i<33 :
		if activeThread < 3:
			# Mulai Thread pekerja baru 
			thread.start_new_thread(populatePart, (i,))
			i=i+1
			activeThread = activeThread + 1
		time.sleep(1)
		timecount = timecount+1
		# Menghitung persentasi keselesaian 
		print((count/(40*32.0)) , populatedData.shape , i , timecount )
	while activeThread > 0:
		time.sleep(5)
		clear();
		timecount = timecount+5
		print((count/(40*32.0)), populatedData.shape , i , timecount)
	
	print(populatedData.shape)

	print('saving data to csv ... ')
	np.savetxt('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded.csv', populatedData, delimiter=',')
