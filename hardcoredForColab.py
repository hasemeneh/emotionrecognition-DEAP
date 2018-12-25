from scipy.io import loadmat
import numpy as np
import sys
import preprocessing as prp
import time
import preprocessing as prp
# import only system from os 
from os import system, name 
from sklearn.decomposition import PCA
pca = PCA(n_components=128)
header =  np.array(['Fp1 Theta','AF3 Theta','F7 Theta','F3 Theta','FC1 Theta','FC5 Theta','T7 Theta','C3 Theta','CP1 Theta','CP5 Theta','P7 Theta','P3 Theta','Pz Theta','PO3 Theta','O1 Theta','Oz Theta','O2 Theta','PO4 Theta','P4 Theta','P8 Theta','CP6 Theta','CP2 Theta','C4 Theta','T8 Theta','FC6 Theta','FC2 Theta','F4 Theta','F8 Theta','AF4 Theta','Fp2 Theta','Fz Theta','Cz Theta','Fp1 Beta','AF3 Beta','F7 Beta','F3 Beta','FC1 Beta','FC5 Beta','T7 Beta','C3 Beta','CP1 Beta','CP5 Beta','P7 Beta','P3 Beta','Pz Beta','PO3 Beta','O1 Beta','Oz Beta','O2 Beta','PO4 Beta','P4 Beta','P8 Beta','CP6 Beta','CP2 Beta','C4 Beta','T8 Beta','FC6 Beta','FC2 Beta','F4 Beta','F8 Beta','AF4 Beta','Fp2 Beta','Fz Beta','Cz Beta','Fp1 Alpha','AF3 Alpha','F7 Alpha','F3 Alpha','FC1 Alpha','FC5 Alpha','T7 Alpha','C3 Alpha','CP1 Alpha','CP5 Alpha','P7 Alpha','P3 Alpha','Pz Alpha','PO3 Alpha','O1 Alpha','Oz Alpha','O2 Alpha','PO4 Alpha','P4 Alpha','P8 Alpha','CP6 Alpha','CP2 Alpha','C4 Alpha','T8 Alpha','FC6 Alpha','FC2 Alpha','F4 Alpha','F8 Alpha','AF4 Alpha','Fp2 Alpha','Fz Alpha','Cz Alpha','Valence','Arousal','Dominance','Liking'])
activeThread = 0
count = 0
populatedData = np.zeros(132)
def populatePart(n):
	data = np.zeros(132)
	label = np.zeros(4)
	arrayX = np.array
	vstackX = np.vstack
	hstackX = np.hstack
	appendX = np.append
	roundX = np.round
	zerosX = np.zeros
	multiplier = np.ones((8064,1))
	matTemp = loadmat('drive/My Drive/skripsoy/dataset/'+str(n)+'.mat')
	print(n)
	labelTemp=matTemp['labels']
	dataTemp = matTemp['data']
	bandPassX =  prp.bandpassX
	stftX =  prp.stft
	for y in range(0,40):
		labelY = labelTemp[y]
		labelY = roundX(labelY)
		labelY = (labelY>=4.5).astype(int)
		dataY = dataTemp[y]
		dataY = dataY[:32].T
		bandPassed = bandPassX(dataY)
		dataY = None
		temporalData = bandPassed[0].T
		temporalData = vstackX((temporalData,bandPassed[1].T))
		temporalData = vstackX((temporalData,bandPassed[2].T))
		temporalData = vstackX((temporalData,bandPassed[3].T))
		bandPassed = None
		temporalData = temporalData.T
		tobeprocessed = arrayX([labelY])
		tobeprocessed = tobeprocessed * multiplier
		temporalData = hstackX((temporalData,tobeprocessed))
		tobeprocessed = None
		data = vstackX((data,temporalData))
		temporalData = None
		print(n , y)
	global populatedData
	populatedData = vstackX((populatedData,data[1:]))
	data = None

if __name__ == '__main__':
	i=1
	timecount = 0
	while i<33 :
		populatePart(i)
		print(i)
		i=i+1
	
	print(populatedData.shape)

	print('saving data to csv ... ')
	np.savetxt('/drive/My Drive/skripsoy/Testing.csv', populatedData, delimiter=',')
