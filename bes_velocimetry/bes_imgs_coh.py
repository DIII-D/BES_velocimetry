import sys
import os
sys.path.append('/home/duxiaodi/xpsi/py_commons')
import hold
import bes_data
import bes_processing
import numpy as np
import scipy.interpolate as si
import matplotlib as mpl
import matplotlib.pylab as plt
import imageio
from scipy.signal import coherence, welch, fftconvolve
#mpl.interactive('t')

# List of PNG file paths
#images = ['image1.png', 'image2.png', 'image3.png']
def generate_gif(fnlist,output_fn1):
    # Combine into a GIF
    output_gif = './'+str(shot)+'/'+output_fn1+'.gif'
    frames = [imageio.v3.imread(img) for img in fnlist]
    imageio.mimsave(output_gif, frames, duration=0.1)  # Adjust duration (in seconds) as needed
    print(f"GIF saved as {output_gif}")
    return
def check():
    plt.subplot(311)
    plt.plot(t, signal1, label="Original Signal 1")
    plt.plot(t, signal2, label="Original Signal 2", alpha=0.7)
    plt.subplot(312)
    plt.plot(f, Cxy, label="Coherence")
    plt.legend()
    plt.subplot(313)
    plt.plot(t, filtered_signal1, label="Filtered Signal 1 (Coherent Part)")
    plt.plot(t, filtered_signal2, label="Filtered Signal 2 (Coherent Part)", alpha=0.7,color='red')
    plt.legend()
    return


if __name__ == '__main__':
   shot = 199968
  # shot = 199972
  # shot = 199975
  # shot = 200729
   if shot == 190850:
      xr = np.array([0.521,0.523])
   if shot == 199968:
      #xr = np.array([0.655,0.657]);cutoff = np.array([31,64])
      xr = np.array([0.631,0.634]);cutoff = np.array([25,55])
      ch_missing = np.array([5,13,33,41,43,51])
   if shot == 199972:
      xr = np.array([0.32,0.322])
      ch_missing = np.array([5,13,33,41,43,51])
      cutoff = np.array([25,125])
   if shot == 199975:
      xr = np.array([0.32,0.322])
      ch_missing = np.array([5,13,33,41,43,51])
      #cutoff = np.array([25,50])
      #cutoff = np.array([25,65])
      cutoff = np.array([75,100])
      xr = np.array([0.32,0.36])
      xr = np.array([0.36,0.365])
      xr = np.array([0.4,0.402])
   if shot == 200729:
      # case 1
      xr = np.array([0.5,0.502])  #GOAL is to run from 0.3 to 2.0, ideally 5.0. Currently broken into 0.002 second blocks.
      cutoff = np.array([100,125]) #cutoff for the bandpass filter, for initial analyses use [30,55] (kHz units). 
      #Generally nothing above 300kHZ, generally 20-30 kHZ chunks are ideal fidelity.
      # case 2
      xr = np.array([0.7,0.702])
      # all taes
      cutoff = np.array([100,150])
      #tae1
      cutoff = np.array([97.2,108.5])
      #tae2
      cutoff = np.array([108.5,115.5])
      #tae3
      cutoff = np.array([113.8,124.5])
      #tae4
      cutoff = np.array([125.9,136.5])
      #tae5
      #cutoff = np.array([136.5,147])
      ch_missing= np.array([2,5,10,13])


   fn_rawdata = 'bes.s'+str(shot)+'.h5'
   output_fn1 = 'imgs.'+str(int(xr[0]*1e3))+'.' \
                           +str(int(xr[1]*1e3))+'.'+fn_rawdata
   print('input: '+ fn_rawdata)
   print('output: '+ output_fn1)
   path = '/home/duxiaodi/xpsi/velocimetry/python/'
   a = bes_data.BES_data.from_h5file(path+fn_rawdata)
   processing = bes_processing.BES_Processing()
   out = processing.rawToPostProcessed(a)
   bes_data.save_h5_from_object(out,output_fn1)


#plot_eddy()




