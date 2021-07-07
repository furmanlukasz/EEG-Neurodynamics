import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat

def loadData(filename):
    data = loadmat(filename)
    electrodes = data['chanlocs']['labels']
    return np.array(data['EEGdata']),[electrodes[0][i][0] for i in range(len(electrodes[0]))]

def SFT(eegSampleRaw):
    f, t, Zxx = signal.stft(eegSampleRaw, fs=250, nperseg=62, noverlap=59)
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=10, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    Zxx = Zxx.T
    # plt.plot(f, np.abs(Zxx[0].real), '-o', label='FFT one row')
    plt.savefig(
        "Spectrogram_SFT" + "s_.png", dpi=250)
    plt.close()
    return


if __name__ == '__main__':
    data = loadData('../DataSets/teData.mat')
    print(data[0][0].shape)
    n = 10
    SFT(data[0][n]) # data[0][n] gdzie n dana elektroda