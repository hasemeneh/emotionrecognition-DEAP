import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

csv = pd.read_csv('data-bandpassed-alpha-beta-gamma-pca-63DetikAllchannel-rounded-T.csv').values
print("done reading data")
csv = csv.T
n_features = 96
n_label = 4


data = csv[:n_features].T
label = csv[n_features:(n_features+n_label)]
# label = label.T

sm = SMOTE(random_state=42)
X_Resampled, y_Resampled = sm.fit_resample(data, label[0])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
np.savetxt('data-bandpassed-alpha-beta-gamma-pca-63DetikAllchannel-rounded-T-smote-arousal.csv', ResampledData, delimiter=',')
# print('Resampled dataset shape %s' % Counter(y_res))
sm = SMOTE(random_state=42)
X_Resampled, y_Resampled = sm.fit_resample(data, label[1])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
np.savetxt('data-bandpassed-alpha-beta-gamma-pca-63DetikAllchannel-rounded-T-smote-valence.csv', ResampledData, delimiter=',')
# print('Resampled dataset shape %s' % Counter(y_res))
sm = SMOTE(random_state=42)
X_Resampled, y_Resampled = sm.fit_resample(data, label[2])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
np.savetxt('data-bandpassed-alpha-beta-gamma-pca-63DetikAllchannel-rounded-T-smote-dominance.csv', ResampledData, delimiter=',')
# print('Resampled dataset shape %s' % Counter(y_res))
sm = SMOTE(random_state=42)
X_Resampled, y_Resampled = sm.fit_resample(data, label[3])
ResampledData = np.hstack((X_Resampled,np.array([y_Resampled]).T))
np.savetxt('data-bandpassed-alpha-beta-gamma-pca-63DetikAllchannel-rounded-T-smote-liking.csv', ResampledData, delimiter=',')
# print('Resampled dataset shape %s' % Counter(y_res))
