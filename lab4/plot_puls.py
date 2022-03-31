from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import butter, lfilter

sample_freq = 40

lower_bound = 30
upper_bound = 240


hpCutoff_freq = lower_bound /60
lpCutoff_freq = upper_bound/60


def raspi_import(path):
    
    data = np.loadtxt(path, delimiter=" ").T

    sample_period = 1 / sample_freq

    return sample_period, data

def butter_coeff(cutoffFreq, sampleFreq, filterType='high', order=6):
    # Find the nyquist frequency
    nyquistFreq = 0.5 * sampleFreq
    # Normalized frequency for cutoff
    normal_cutoff = cutoffFreq / nyquistFreq
    return signal.butter(order, normal_cutoff, btype=filterType, analog=False)


def butter_filter(dataPoints, cutoffFreq, sampleFreq, filterType='high', order=6):
    b, a = butter_coeff(cutoffFreq, sampleFreq, filterType, order=order)
    return signal.filtfilt(b, a, dataPoints)


def window(data, window="hamming"):
    data = data *signal.get_window(window,data.shape[1])
    return data


# Import data from bin file
sample_period, data = raspi_import('Simen_T_2.bin')

data = signal.detrend(data) # removes DC component for each channel
#sample_period *= 1e-6  # change unit to micro seconds
data = butter_filter(data, hpCutoff_freq, sample_freq)
data = butter_filter(data, lpCutoff_freq, sample_freq, 'low')


data = window(data)
# Generate time axis
num_of_samples = data.shape[1]  # returns shape of matrix
t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)

# Generate frequency axis and take FFT
freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
spectrum = np.fft.fft(data)  # takes FFT of all channels
#spectrum = window(spectrum)

puls = []
for spec in spectrum :
    puls.append(abs(freq[np.abs(spec).argmax()]*60))

print(puls)






def snr(color):
    sig = np.amax(color)
    sig_index = np.where(color == np.amax(color))
    #legge til nabo til max i signal fordi naboene er også en del av signalet. 
    color[sig_index] == 0
    noise = np.mean(color)
    snr = sig-noise
    return snr
print("SNR FOR COLORS : ")
print(snr((20*np.log10(np.abs(spectrum[0].T)[:int((len(freq)/8))]))))
print(snr((20*np.log10(np.abs(spectrum[1].T)[:int((len(freq)/8))]))))
print(snr((20*np.log10(np.abs(spectrum[2].T)[:int((len(freq)/8))]))))

def plot_time():
    plt.figure("Data")
    plt.title("Time domain signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage")
    plt.plot(t, data[0], "r", label="R")
    plt.plot(t, data[1], "g", label="G")
    plt.plot(t, data[2], "b", label="B")
    plt.legend()
    plt.show()
    return

plot_time()

def plot_freq():
    plt.title("Power spectrum of signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Relative Power [dB]")
    plt.plot(freq[:(int(len(freq)/8))], 20*np.log10(np.abs(spectrum[0].T)[:int((len(freq)/8))]), "r", label="R") # get the power spectrum
    plt.plot(freq[:int((len(freq)/8))], 20*np.log10(np.abs(spectrum[1].T)[:int((len(freq)/8))]), "g", label="G") # get the power spectrum
    plt.plot(freq[:int((len(freq)/8))], 20*np.log10(np.abs(spectrum[2].T)[:int((len(freq)/8))]), "b", label="B") 
    plt.legend()
    plt.show()
    return
plot_freq()










# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
''' plt.subplot(2, 1, 1)
plt.title("Time domain signal")
plt.xlabel("Time [us]")
plt.ylabel("Voltage")
plt.plot(t, 3.3/(2**12)*data)

plt.subplot(2, 1, 2)
plt.title("Power spectrum of signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.plot(freq, 20*np.log10(np.abs(spectrum))) # get the power spectrum

plt.show() '''
