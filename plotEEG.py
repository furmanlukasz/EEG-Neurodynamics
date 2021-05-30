
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.fftpack
from scipy.io import loadmat
import h5py

def plotSignal(name, data):
  fig,ax = plt.subplots(1,2,figsize=(20,6))
  ax[0].plot(time,data)
  ax[0].set_xlabel('Time (s)')
  ax[0].set_title('Time domain'+ name)

  dataX = scipy.fftpack.fft(data/npnts)
  ax[1].plot(hz,np.abs(dataX[:len(hz)]))
  ax[1].set_xlim([0,60])
  ax[1].set_xlabel('Frequency (Hz)')
  ax[1].set_title('Frequency domain' + name)

def notch(f,data,srate):
    f0 = f
    Q = f
    b,a = signal.iirnotch(f0,Q,srate)
    #freq,h = signal.freqz(b,a,fs=srate)
    notched = signal.filtfilt(b, a, data)
    return notched

if __name__ == '__main__':
    f = h5py.File('Data_fl.mat')
    df = np.array(f['y']).T
    srate = 256
    el_num = 5


    in_r ,out_r = 3000,15000
    time = np.arange(len(df[0][in_r:out_r]))
    npnts = len(time)
    hz = np.linspace(0, srate / 2, int(npnts / 2) + 1)
    for i in range(9):
        data2plot = notch(51,np.array(df[i][in_r:out_r]),srate)
        plotSignal('electrode'+str(i),np.array(data2plot))
        plt.savefig('ele_'+str(i)+'.png')


def plotingtest():
    img = np.random.randint(-100, 100, (10, 10))
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(6, 6, hspace=0.6, wspace=0.6)
    main_ax = fig.add_subplot(grid[:-1, 1:])

    mm = main_ax.imshow(img)
    cbar_ax = fig.add_axes([0.007, 0.2, 0.03, 0.7])
    fig.colorbar(mm, cax=cbar_ax)
    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

    plt.show()