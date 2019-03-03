import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE

csv = pd.read_csv('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded.csv').values
print("done reading data")
csv = csv.T
n_features = 128
n_label = 4


data = csv[:n_features].T
label = csv[n_features:(n_features+n_label)]
# label = label.T

sm = SMOTE(random_state=42)
X_Resampled, y_Resampled = sm.fit_resample(data, label[0])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
ResampledData = np.vstack((np.zeros((1, n_features+1)),ResampledData))
np.savetxt('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded-smote-arousal.csv', ResampledData, delimiter=',')
# print('Resampled dataset shape %s' % Counter(y_res))
sm = SMOTE(random_state=42)
X_Resampled, y_Resampled = sm.fit_resample(data, label[1])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
ResampledData = np.vstack((np.zeros((1, n_features+1)),ResampledData))
np.savetxt('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded-smote-valence.csv', ResampledData, delimiter=',')
# print('Resampled dataset shape %s' % Counter(y_res))
sm = SMOTE(random_state=42)
X_Resampled, y_Resampled = sm.fit_resample(data, label[2])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
ResampledData = np.vstack((np.zeros((1, n_features+1)),ResampledData))
np.savetxt('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded-smote-dominance.csv', ResampledData, delimiter=',')
# print('Resampled dataset shape %s' % Counter(y_res))
sm = SMOTE(random_state=42)
X_Resampled, y_Resampled = sm.fit_resample(data, label[3])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
ResampledData = np.vstack((np.zeros((1, n_features+1)),ResampledData))
np.savetxt('data-bandpassed-theta-alpha-beta-gamma-pca-60DetikAllchannel-rounded-smote-liking.csv', ResampledData, delimiter=',')
# print('Resampled dataset shape %s' % Counter(y_res))
