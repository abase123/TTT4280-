from ossaudiodev import SNDCTL_DSP_RESET
from statistics import mean
from xml.etree.ElementTree import QName
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.signal as signal
from scipy.fft import fft , ifft

def raspi_import(path, channels=5):
    """
    Import data produced using adc_sampler.c.
    Returns sample period and ndarray with one column per channel.
    Sampled data for each channel, in dimensions NUM_SAMPLES x NUM_CHANNELS.
    """

    with open(path, 'r') as fid:
        sample_period = np.fromfile(fid, count=1, dtype=float)[0]
        data = np.fromfile(fid, dtype=np.uint16)
        data = data.reshape((-1, channels))
        #removing garbage signals

        data = data.T
    return sample_period, np.array(data)



# Import data from bin file
sample_period, data = raspi_import('bil_bak_1.bin')


#data = signal.detrend(data, axis=0)  # removes DC component for each channel
sample_period *= 1e-6  # change unit to micro seconds


# Generate time axis
num_of_samples = data.shape[1]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels

data = signal.detrend(data)

I = data[4]
Q = data[2]


window = np.hamming(len(Q))

X_f = np.fft.fft(I+1j*Q)
X_f_win = window * X_f

dopplershift = freq[np.abs(X_f_win).argmax()] #int(freq[np.where(spectrum == max(spectrum))])
print(f"Dopplershift is : {dopplershift} Hz and the velocity is {dopplershift/160} m/s")
# SNR FOR WINDOW
#peak = np.abs(X_f).argmax()

#nois_floor = np.abs(np.mean(X_f[0:2200]))

#SNR = peak - nois_floor

#print(SNR)


d_f = 1/(sample_period)





# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
plt.subplot(2, 1, 1)
#plt.title("Time domain signal")
#plt.xlabel("Time [us]")
#plt.ylabel("Voltage")
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq,X_f_win)

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq,20*np.log10(np.abs(X_f)))
#plt.plot(freq, 20*np.log10(np.abs(spectrum))) # get the power spectrum

plt.show()


