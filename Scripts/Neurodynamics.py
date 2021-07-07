from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.fftpack
import h5py
from scipy.io import loadmat
from scipy.signal import welch
from scipy.signal import spectrogram
from sklearn.preprocessing import normalize as nr
import matplotlib.ticker as mticker
from pyrqa.image_generator import ImageGenerator
from pyrqa.computation import RPComputation
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import FixedRadius, RadiusCorridor
from pyrqa.metric import EuclideanMetric
from pyrqa.metric import TaxicabMetric
from pyrqa.metric import Sigmoid
from pyrqa.metric import Cosine
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import Unthresholded
from basic_units import secs, hertz, minutes
from nolitsa import dimension
import pywt
import pickle
import math

def loadData(filename):
    data = loadmat(filename)
    electrodes = data['chanlocs']['labels']
    return np.array(data['EEGdata']),[electrodes[0][i][0] for i in range(len(electrodes[0]))]

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def norm(arr, td, emb):
    tdemb = td * emb
    nslice = int(len(arr) / tdemb)
    array_size = (int(len(arr) / td), 1)
    temp = np.zeros(array_size)
    for i in range(nslice):

        j = i * tdemb
        nor = 0
        m = i * emb

        for k in range(emb):
            nor += (arr[j + k * td]) * (arr[j + k * td])

        nor = math.sqrt(nor)

        for l in range(emb):
            temp[int(m + l)] = arr[l * td + j]/nor
    return temp


class ComputeTask(object):

    def __init__(self, subject,data,srate,electrode):
        # data[0] eeg arrays
        # data[1] electrode names

        self.data = data
        self.eegData = self.data[0]

        self.srate = srate
        self.electrode = electrode
        self.el_index = self.data[1].index(self.electrode)
        self.subject = subject
        self.interval = len(self.eegData[self.el_index])
        self.timestep = self.interval / self.srate
        self.eegSample = self.eegData[self.el_index]
        self.eegSampleRaw = self.eegSample
        self.time = np.arange(len(self.eegSample))/self.srate

        self.regular_info_title = 'Electrode: '+ self.electrode + ', sub: '+ self.subject + ', timestamp: ' + str(self.timestep)

    def getChanLenght(self):
        return len(self.eegSample)

    def normDataSlice(self,npoints,td,emb):
        self.eegSample = signal.resample(norm(self.eegSample,td,emb), npoints)

    def testSignal(self,freq,duration):

        x = np.arange(duration)  # the points on the x axis for plotting
        # compute the value (amplitude) of the sin wave at the for each sample
        y = np.sin(2 * np.pi * freq * (x / self.srate))
        y1 = np.sin(2 * np.pi * 20 * (x / self.srate))

        y1[:len(y1)-(len(y1)-int(duration/3))] = y[:len(y1)-(len(y1)-int(duration/3))]
        y1[int(2*(duration/3)):] = y[int(2*(duration/3)):]
        sig = y1
        self.eegSample = sig
        self.eegSampleRaw = sig
        return self.eegSample #normalize(self.eegSample, -1, 1)

    def setDataSlice(self,slice, counter=1):
        self.eegSample = np.array(slice)
        self.counter = counter
        self.interval = len(slice)
        self.timestep = (self.interval / self.srate) * self.counter
        self.time = np.arange(len(self.eegSample)) / self.srate
        self.eegSampleRaw = self.eegSample
        self.regular_info_title = 'Electrode: ' + self.electrode + ', sub: ' + self.subject + ', timestamp: ' + str(
            self.timestep)

    def getDataSlice(self, fromrange, torange,counter=1):

        ndata = np.zeros((len(self.eegData), torange - fromrange))
        for i in range(len(self.eegData)):
            ndata[i] = self.eegData[i][fromrange:torange]
        self.counter = counter
        self.eegData = ndata
        self.interval = len(self.eegData[self.el_index])
        self.timestep = (self.interval / self.srate)*self.counter
        self.eegSample = self.eegData[self.el_index]
        self.eegSampleRaw = self.eegSample
        self.time = np.arange(len(self.eegSample)) / self.srate

        self.regular_info_title = 'Electrode: ' + self.electrode + ', sub: ' + self.subject + ', timestamp: ' + str(self.timestep)
        return self.eegSample



    def computeRP(self,td,emb,metric):
        self.timedelay = td
        self.embedding = emb
        self.metric = metric


        time_series = TimeSeries(self.eegSample,
                                 embedding_dimension=self.embedding,
                                 time_delay=self.timedelay)

        settings = Settings(time_series,
                            analysis_type=Classic,
                            neighbourhood=Unthresholded(),
                            similarity_measure=eval(self.metric),  # Sigmoid,#TaxicabMetric,#Sigmoid,#EuclideanMetric,
                            theiler_corrector=1)

        computation = RPComputation.create(settings,
                                           verbose=True)
        result = computation.run()
        self.recurrence_matrix_reverse = result.recurrence_matrix_reverse
        return result.recurrence_matrix_reverse


    def SaveRecurencePlot(self, title,matrix='none'):

        if isinstance(matrix, str):
            matrix = self.recurrence_matrix_reverse

        fig, axes = plt.subplots(2, 1,figsize=(10,14),gridspec_kw={'height_ratios': [4, 1],}) #(6,9)
        fig.subplots_adjust(hspace=0.05)

        IMGMM = axes[0].imshow(matrix[::-1], cmap='jet', interpolation='none', origin='upper',extent=[0,int((self.interval*4)),int((self.interval*4)),0])
        axes[0].locator_params(axis="x", nbins=10)
        axes[0].locator_params(axis="y", nbins=10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        #axes[0].set_xlabel('samples',fontsize=17)
        axes[0].set_ylabel('time(ms)',fontsize=19)

        axes[0].invert_yaxis()
        cbar_ax = fig.add_axes([0.55, 0.980, 0.43, 0.015])
        fig.colorbar(IMGMM,cax=cbar_ax,orientation="horizontal")

        axes[1].plot(self.time, self.eegSampleRaw)
        axes[1].set_xlabel('time(ms)',fontsize=19)
        top_side = axes[1].spines["top"]
        top_side.set_visible(False)

        right_side = axes[1].spines["right"]
        right_side.set_visible(False)
        axes[1].autoscale(enable=True, axis='both', tight=True)
        plt.xticks(fontsize=12)
        plt.tight_layout()

        title_string = self.electrode + ', ' + 'emb = ' + str(self.embedding) + ' td = ' + str(self.timedelay)
        subtitle_string = 'sub: '+ self.subject + ' timestamp ' + str(self.timestep)
        plt.suptitle(title_string,x=0.18, y=0.99, fontsize=18)
        plt.title(subtitle_string,x=-0.82, y=-2.0,fontsize=14)

        plt.savefig(
            "../RR_plots/Dist_" + self.subject + '_electrode_' + self.electrode + '_' + '_emb_' + str(
                self.embedding) + '_td_' + str(self.timedelay) + '_tstamp_' + str(self.timestep) + 's_'+title+'.png', dpi=250)
        plt.close()

    def SaveTimeFreqDomain(self):

        data = np.array(self.eegSampleRaw)
        time = np.arange(len(data))
        npnts = len(time)
        hz = np.linspace(0, self.srate / 2, int(npnts / 2) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        ax[0].plot(time/self.srate, data)
        #ax[0].set_ylim([0.98, 1.09])
        ax[0].set_xlabel('Time (s)')
        ax[0].set_title('Time domain')
        ax[0].autoscale(enable=True, axis='both', tight=True)

        dataX = scipy.fftpack.fft(data / npnts)
        ax[1].plot(hz, np.abs(dataX[:len(hz)]))
        ax[1].set_xlim([0, 50])
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_title('Frequency domain')

        plt.suptitle(self.regular_info_title,x=0.50, y=0.99, fontsize=18)

        plt.savefig(
            "../TimeDomain_plots/TimeDomainAndFFT_" + self.subject + '_electrode_' + self.electrode + '_'  '_tstamp_' + str(self.timestep) + 's_.png', dpi=250)
        plt.close()

    def SaveFreqDomain(self):

        data = np.array(self.eegSampleRaw)
        time = np.arange(len(data))
        npnts = len(time)
        dynamicHz, dynamicX1 = welch(data, nfft=len(data))

        hz = np.linspace(0, self.srate / 2, int(npnts / 2) + 1)
        fig, ax = plt.subplots(figsize=(10, 4))

        dataX = scipy.fftpack.fft(data / npnts)
        ax.plot(hz, np.abs(dynamicX1[:len(dynamicHz)]), '-o', label='FFT Welch Method')
        #ax.plot(hz, np.abs(dataX[:len(hz)]), '-o', label='FFT Normal')
        ax.set_xlim([0, 50])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title('Frequency domain')

        plt.suptitle(self.regular_info_title,x=0.50, y=0.99, fontsize=18)

        plt.savefig(
            "../TimeDomain_plots/TimeDomainAndFFT_" + self.subject + '_electrode_' + self.electrode + '_'  '_tstamp_' + str(self.timestep) + 's_.png', dpi=250)
        plt.close()

    def SFT(self):
        f, t, Zxx = signal.stft(self.eegSampleRaw,fs=250, nperseg=62,noverlap=61)
        plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=10, shading='gouraud')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        Zxx = Zxx.T
        #plt.plot(f, np.abs(Zxx[0].real), '-o', label='FFT one row')
        plt.savefig(
            "../Spectrogram_plots/Spectrogram_SFT" + self.subject + '_electrode_' + self.electrode + '_'  '_tstamp_' + str(
                self.timestep) + 's_.png', dpi=250)
        plt.close()
        return

    def SaveSpectrogram(self):
        f, t, Sxx = spectrogram(np.array(self.eegSampleRaw), fs=250) #noverlap=10,
        plt.pcolormesh(t, f[0:50], Sxx[0:50], shading='gouraud')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title(self.regular_info_title)
        plt.savefig(
            "../Spectrogram_plots/Spectrogram_" + self.subject + '_electrode_' + self.electrode + '_'  '_tstamp_' + str(
                self.timestep) + 's_.png', dpi=250)
        plt.close()

# example usage
# if __name__ == '__main__':
#
#     task1 = ComputeTask('testSubject','../DataSets/teData.mat',250,'O1')
#     task1.getDataSlice(2000,3000)
#     task1.SaveTimeFreqDomain()
#     task1.normDataSlice()
#     task1.computeRQA(td=2,emb=10,metric='Cosine')
#     task1.SaveRecurencePlot()