from fileinput import filename
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

c=343.4
d = 0.062
a =2.6846

print((32000*d)/c)
mic_1 = np.array([0,1])*a
mic_2 =np.array([-np.sqrt(3)/2,-0.5])*a
mic_3 =np.array([np.sqrt(3)/2,-0.5])*a

mic_matrix=np.array([mic_2-mic_1, mic_3-mic_1, mic_3-mic_2])

#test lag
t_matrix=0.003*np.array([0.386367 , -0.087589 , -0.473957])

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
        
        
         #DC-remove
        data = signal.detrend(data, axis=0) 
       
        
    return sample_period, data

# Import data from bin file
#sample_period, data = raspi_import('data.bin')
#data = signal.detrend(data, axis=0)  # removes DC component for each channel
#sample_period *= 1e-6  # change unit to micro seconds
# Generate time axis
#num_of_samples = data.shape[0]  # returns shape of matrix
##t = np.linspace(start=0, stop=num_of_samples*sample_period, num=num_of_samples)
# Generate frequency axis and take FFT
#freq = np.fft.fftfreq(n=num_of_samples, d=sample_period)
#spectrum = np.fft.fft(data, axis=0)  # takes FFT of all channels




def corralation(data1,data2):
   corr = np.correlate(data1[6 :-6], data2)
   lags = np.linspace(int(-len(corr) / 2), int(len(corr) / 2), num=len(corr))
   l_max=int(lags[np.where(corr == max(corr))])

   return corr,lags,l_max
 
 
def autocorr(data1):
     corr,lags,l_max=corralation(data1,data1)
     plt.plot(lags,corr)
     plt.axvline(x=0.0,color="black",linestyle='--')
     plt.xlabel("lags")
     plt.ylabel("corr")
     plt.show()
     return
     




def find_angle(t_matrix,mic_matrix):
    x_vec = (-c * (np.linalg.pinv(mic_matrix) @ t_matrix))
    angle = np.arctan2(x_vec[1],x_vec[0])
    ## numpy degree , bruk det på arctan 
    angle = (angle*180)/np.pi 
    if angle < 0:
        angle=angle+360
        
        
    return angle 




def cal_angle_all(filname):
    
    sample_period,data = raspi_import(filname)
    sample_freq = 1/sample_period
    autocorr(data[:,0])
    
    xcorr21, lag21 , l_max21 = corralation(data[:,1],data[:,0])
    xcorr31, lag31,   l_max31  = corralation(data[:,2],data[:,0])
    xcorr32, lag32 ,  l_max32 = corralation(data[:,2],data[:,1])
    
    
    return find_angle( [l_max21,l_max31,l_max32], mic_matrix)

# Plot the results in two subplots
# NOTICE: This lazily plots the entire matrixes. All the channels will be put into the same plots.
# If you want a single channel, use data[:,n] to get channel n
#plt.subplot(2, 1, 1)
#plt.xlabel("Time [us]")
#plt.ylabel("Voltage")
#plt.plot(t, 3.3/(2**12)*data)

#plt.subplot(2, 1, 2)
#plt.title("Power spectrum of signal")
#plt.xlabel("Frequency [Hz]")
#plt.ylabel("Power [dB]")
#plt.plot(freq, 20*np.log10(np.abs(spectrum))) # get the power spectrum

#plt.show()


def std_90(l):
    var=0
    for i in l : 
        var+=((i-90.0)**2)
        std=(var/8)**0.5
    return std

def std_315(l):
    var=0
    for i in l : 
        var+=((i-315.0)**2)
        std=(var/5)**0.5
    return std

def std_135(l):
    var=0
    for i in l : 
        var+=((i-135.0)**2)
        std=(var/8)**0.5
    return std

print(cal_angle_all("135_8.bin"))

l_90=[96.6 , 87.0 , 83.4 , 90.0 , 90.0 , 90.0 , 87.0 , 90.0]

l_315=[330.0 , 338.2 , 330.0 , 326.3 , 326.32 ]
l_135 = [126.6 , 132.2 , 126.6 , 127.6, 132.2 , 126.6 , 126.6 , 132.2 ]

print(std_90(l_90))
print(std_315(l_315))
print(std_135(l_135))