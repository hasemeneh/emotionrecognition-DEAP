import csv
import time

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal 
from scipy.signal import butter, lfilter,cwt,ricker
from scipy.fftpack import rfft, irfft, fftfreq
import pywt

header = ['Timestamp', 'F3 Value','FC5 Value','F7 Value','T7 Value','P7 Value','O1 Value','O2 Value','P8 Value','T8 Value','F8 Value','AF4 Value','FC6 Value','F4 Value','AF3 Value']
vstack = np.vstack

def fft(data, have_header=True):
	ibp=np.zeros(len(data[0]))
	for column in data:
		fft = scipy.fft(column)
		# for j in range(0, len(fft)):
		# 	if j>=10:fft[j]=0
		# np.array()
		iffted = scipy.ifft(fft)
		ibp = vstack((ibp, iffted))
	data = None
	ibp = ibp[1:]
	return ibp
def stft(data, have_header=True):
	ibp=np.zeros(len(data[0]))
	fs = 128
	for column in data:
		maxVal = np.amax(column)
		minVal = np.amin(column)
		amp = abs(maxVal - minVal)/2
		f, t, Zxx = signal.stft(column, fs)
		# for j in range(0, len(fft)):
		# 	if j>=10:fft[j]=0
		# np.array()
		Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
		_, xrec = signal.istft(Zxx, fs)
		# iffted = istft(fft)
		ibp = vstack((ibp, xrec))
	data = None
	ibp = ibp[1:]
	return ibp
def dwt(data, have_header=True):
	ibp1=np.zeros(len(data[0])/2)
	ibp2=np.zeros(len(data[0])/2)
	
	for column in data:
		# fft = scipy.fft(column)
		# # for j in range(0, len(fft)):
		# # 	if j>=10:fft[j]=0
		# # np.array()
		# iffted = scipy.ifft(fft)
		# cwtmatr = cwt(column, ricker, 128)
		(cA, cD) = pywt.dwt(column, 'db1')
		ibp1 = vstack((ibp1, cA))
		ibp2 = vstack((ibp2, cD))
	data = None
	ibp1 = ibp1[1:]
	ibp2 = ibp2[1:]
	return ibp1 , ibp2

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

def bandpass(data, have_header=True):
	bpass = np.zeros(len(data.T[0])-1)
	bpassABG = np.zeros(len(data.T[0])-1)
	cutoff=[1,4,8,12,30]
	#print len(data.T[0])	
	for column in data.T:
		bps = butter_bandpass_filter(column[have_header:],1,30,128,2)
		bpass= vstack((bpass, np.array(bps)))
	# for x in range(0,4):
	# 	bpastemp = butter_bandpass_filter(data.T[1][have_header:],cutoff[x],cutoff[x+1],128,2)
	# 	if(np.array_equal(bpassABG,np.zeros(len(column)-1))):
	# 		bpassABG = np.array(bpastemp)
	# 	else:
	# 		bpassABG= vstack((bpassABG,np.array(bpastemp)))
	bpass = bpass[1:].T
	bpassABG = bpassABG.T
	return bpass, bpassABG

def bandpassX(data, have_header=True):
	data = data.T
	bpass = np.zeros(len(data[0]))
	bpassABG = []
	cutoff=[1,4,8,12,30]
	#print len(data.T[0])	
	# for column in data:
	# 	bps = butter_bandpass_filter(column[have_header:],1,30,128,2)
	# 	bpass= np.vstack((bpass, np.array(bps)))
	
	# bpassX = np.zeros(len(data[0])-1)
	# for column in data:
	# 	bps = butter_bandpass_filter(column[have_header:],0,4,128,2)
	# 	bpassX= np.vstack((bpassX, np.array(bps)))
	# bpassABG.append(bpassX[1:].T)
	bpassX = np.zeros(len(data[0]))
	for column in data:
		bps = butter_bandpass_filter(column,4,7,128,2)
		bpassX= np.vstack((bpassX, np.array(bps)))
	bpassABG.append(bpassX[1:].T)
	bpassX = np.zeros(len(data[0]))
	for column in data:
		bps = butter_bandpass_filter(column,8,15,128,2)
		bpassX= np.vstack((bpassX, np.array(bps)))
	bpassABG.append(bpassX[1:].T)
	bpassX = np.zeros(len(data[0]))
	for column in data:
		bps = butter_bandpass_filter(column,16,31,128,2)
		bpassX= np.vstack((bpassX, np.array(bps)))
	bpassABG.append(bpassX[1:].T)	
	bpassX = np.zeros(len(data[0]))
	for column in data:
		bps = butter_bandpass_filter(column,31,45,128,2)
		bpassX= np.vstack((bpassX, np.array(bps)))
	bpassABG.append(bpassX[1:].T)	
	# for x in range(0,4):	bpassABG.append(bpassX[1:].T)

	# 	bpastemp = butter_bandpass_filter(data.T[1][have_header:],cutoff[x],cutoff[x+1],128,2)
	# 	if(np.array_equal(bpassABG,np.zeros(len(column)-1))):
	# 		bpassABG = np.array(bpastemp)
	# 	else:
	# 		bpassABG= vstack((bpassABG,.arra(bpastempy)))
	# bpass = bpass[1:].Tnp
	# bpassABG = bpassABG.T
	return bpassABG

