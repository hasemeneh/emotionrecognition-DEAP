from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=32)
x = loadmat('1.mat')
data = np.zeros(32)
label = np.zeros(4)
vstackX = np.vstack
fitX=pca.fit
for i in xrange(1,33):
	x = loadmat(str(i)+'.mat')
	print(i)
	for y in x['data']:
		temp = y[:32].T
		fitX(temp)
		data = vstackX((data,pca.singular_values_))

	# data = np.append(data,x['data'],axis=0)	
	label = vstackX((label,x['labels']))
# hasil = pca.fit(data[0])
# label = (x['labels'].astype(int)>=5).astype(int)
data = data[1:]
label = label[1:]
label = np.round(label)
label = (label.astype(int)>=5).astype(int)
print(data.shape)
print(label.shape)
np.savetxt("data.csv", data, delimiter=",")
np.savetxt("label.csv", label, delimiter=",")