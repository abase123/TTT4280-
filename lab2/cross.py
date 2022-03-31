
from random import random
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fft

period=1/((2*np.pi/10))
f_s=100
phase = 10
t = np.arange(0,100,1/f_s) # number of samples
sig =np.sin(period*t) # sin signal

white_noise = np.random.normal(0,100) # making sin signal mixed with random noise

phase_shift_sig=np.sin(period*t+phase) #phase shifted sinus signal
corr = signal.correlate(white_noise,sig,"full")# correlation between the two signals
lags =signal.correlation_lags(len(sig),len(phase_shift_sig)) # lags axises, this lines give us also the negative lag values
corr= corr/np.max(corr) # normalization of corr function

''' FINDING MAX CORR VALUE AND TIME DELAY  '''

max_value = np.amax(corr)
pos_max_corr=np.where(corr==max_value)
pos_max_corr=np.array(pos_max_corr) #sometimes we get two more then one position


center = (len(corr)/2)-1
phase = np.abs(center-pos_max_corr)
print(phase)

print(pos_max_corr)
delta_t = pos_max_corr/f_s
print(delta_t)


                

''' plotting and decoration '''

fig ,(ax_orgin , ax_noise , ax_corr) = plt.subplots(3,1,figsize=(4.8,4.8))
ax_orgin.plot(sig)
ax_orgin.set_title('Original signal')
ax_orgin.set_xlabel('Sample Number')

ax_noise.plot(white_noise)
ax_noise.set_title('Phase shifted signal')
ax_noise.set_xlabel('Sample Number')

ax_corr.plot(lags,corr)
ax_corr.set_title('Cross-correlated signal')
ax_corr.set_xlabel('Lag')

ax_orgin.margins(0, 0.1)
ax_noise.margins(0, 0.1)
ax_corr.margins(0, 0.1)

fig.tight_layout()


plt.show()





