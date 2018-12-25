from scipy.io import loadmat
import numpy as np
import preprocessing as prp
import thread
import time
import preprocessing as prp
# import only system from os 
from os import system, name 
from sklearn.decomposition import PCA
pca = PCA(n_components=128)
matTemp = loadmat(str(1)+'.mat')
labelTemp=matTemp['labels']
dataTemp = matTemp['data']
bandPassX =  prp.bandpassX
arrayX = np.array
vstackX = np.vstack
hstackX = np.hstack
appendX = np.append
roundX = np.round
zerosX = np.zeros
dataY = dataTemp[0]
dataY = dataY[:32].T
np.savetxt('data-original.csv', dataY, delimiter=',')
bandPassed = bandPassX(dataY)
temporalData = bandPassed[0].T
temporalData = vstackX((temporalData,bandPassed[1].T))
temporalData = vstackX((temporalData,bandPassed[2].T))
temporalData = vstackX((temporalData,bandPassed[3].T))
temporalData = temporalData.T
np.savetxt('data-bandpassed.csv', temporalData, delimiter=',')
tempPCA = pca.fit(temporalData).singular_values_
np.savetxt('data-PCAed.csv', tempPCA, delimiter=',')

