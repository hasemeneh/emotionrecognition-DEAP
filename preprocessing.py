import csv
import time
import numpy as np
import scipy
from scipy import signal 
from scipy.signal import butter, lfilter
vstack = np.vstack

def butter_bandpass_filter(data, lowcut, highcut, sampleRate, order=2):
	b, a = butter_bandpass(lowcut, highcut, sampleRate, order=order)
	y = lfilter(b, a, data.astype(np.float))
	return y

def butter_bandpass(lowcut, highcut, sampleRate, order = 2):
	nyq = 0.5 * sampleRate
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype = 'band')
	return b, a


def bandpassX(data, have_header=True):
	data = data.T
	bpass = np.zeros(len(data[0]))
	bpassABG = []
	bpassX = np.zeros(len(data[0]))
# theta
	for column in data:
		bps = butter_bandpass_filter(column,4,7,128,2)
		bpassX= np.vstack((bpassX, np.array(bps)))
	bpassABG.append(bpassX[1:].T)
	bpassX = np.zeros(len(data[0]))
# alpha
	for column in data:
		bps = butter_bandpass_filter(column,8,15,128,2)
		bpassX= np.vstack((bpassX, np.array(bps)))
	bpassABG.append(bpassX[1:].T)
	bpassX = np.zeros(len(data[0]))
# beta
	for column in data:
		bps = butter_bandpass_filter(column,16,31,128,2)
		bpassX= np.vstack((bpassX, np.array(bps)))
	bpassABG.append(bpassX[1:].T)	
	bpassX = np.zeros(len(data[0]))
# Gamma
	for column in data:
		bps = butter_bandpass_filter(column,31,45,128,2)
		bpassX= np.vstack((bpassX, np.array(bps)))
	bpassABG.append(bpassX[1:].T)	
	return bpassABG

