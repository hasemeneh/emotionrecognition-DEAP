import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

csv = pd.read_csv('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded.csv').values
csv = csv.T
n_features = 128
n_label = 4


data = csv[:n_features].T
label = csv[n_features:(n_features+n_label)]
# oversampling data untuk model arousal
sm = SMOTE(random_state=42,kind='borderline1',m_neighbors=20)
X_Resampled, y_Resampled = sm.fit_resample(data, label[0])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
ResampledData = np.vstack((np.zeros((1, n_features+1)),ResampledData))
np.savetxt('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded-smote-arousal.csv', ResampledData, delimiter=',')
# oversampling data untuk model valence
sm = SMOTE(random_state=42,kind='borderline1',m_neighbors=20)
X_Resampled, y_Resampled = sm.fit_resample(data, label[1])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
ResampledData = np.vstack((np.zeros((1, n_features+1)),ResampledData))
np.savetxt('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded-smote-valence.csv', ResampledData, delimiter=',')
print("done reading data")
