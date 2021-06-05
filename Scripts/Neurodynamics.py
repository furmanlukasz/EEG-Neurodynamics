from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.fftpack
import h5py
from scipy.io import loadmat
from scipy.signal import welch
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
        self.time = np.arange(len(self.eegSample))/self.srate

    def getDataSlice(self, fromrange, torange):

        ndata = np.zeros((len(self.eegData), torange - fromrange))
        for i in range(len(self.eegData)):
            ndata[i] = self.eegData[i][fromrange:torange]

        self.eegData = ndata
        self.interval = len(self.eegData[self.el_index])
        self.timestep = self.interval / self.srate
        self.eegSample = self.eegData[self.el_index]
        self.time = np.arange(len(self.eegSample)) / self.srate
    def normDataSlice(self):
        self.eegSample = normalize(self.eegSample,-1, 1)

    def computeRQA(self,td,emb,metric):
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

    def SaveRecurencePlot(self, matrix='none'):

        if isinstance(matrix,str):
            matrix = self.recurrence_matrix_reverse


        fig = plt.figure(figsize=(8, 8))
        grid = plt.GridSpec(6, 6, hspace=0.8, wspace=0.8)
        plt.title(
            self.electrode + ', ' + 'emb = ' + str(self.embedding) + ' td = ' + str(
                self.timedelay) + ' timestamp ' + str(self.timestep),horizontalalignment='left')

        plt.axis('off')
        main_ax = fig.add_subplot(grid[:-1, 1:])

        x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[])
        #y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
        main = main_ax.imshow(matrix[::-1], cmap='jet', interpolation='none',origin='upper')
        main_ax.set_xlabel('samples')
        main_ax.set_ylabel('samples')

        x_hist.plot(self.time,self.eegSample, color='gray')
        x_hist.autoscale(enable=True, axis='both', tight=True)

        x_hist.set_xlabel('time(s)')
        cbar_ax = fig.add_axes([0.91, 0.25, 0.03, 0.63])
        fig.colorbar(main, cax=cbar_ax)
        main_ax.invert_yaxis()

        plt.savefig(
            "../RR_plots/Dist_" + self.subject + '_electrode_' + self.electrode + '_' + '_emb_' + str(
                self.embedding) + '_td_' + str(self.timedelay) + '_tstamp_' + str(self.timestep) + 's_.png', dpi=250)

    def SaveTimeFreqDomain(self):

        data = self.eegSample
        time = np.arange(len(data))
        npnts = len(time)
        hz = np.linspace(0, self.srate / 2, int(npnts / 2) + 1)
        fig, ax = plt.subplots(1, 2, figsize=(20, 6))
        ax[0].plot(time/self.srate, data)
        ax[0].set_xlabel('Time (s)')
        ax[0].set_title('Time domain')


        dataX = scipy.fftpack.fft(data / npnts)
        ax[1].plot(hz, np.abs(dataX[:len(hz)]))
        ax[1].set_xlim([0, 50])
        ax[1].set_xlabel('Frequency (Hz)')
        ax[1].set_title('Frequency domain')
        ax[1].text(51, 0, 'Electrode ' + self.electrode, fontsize=15, color='black')

        plt.savefig(
            "../TimeDomain_plots/TimeDomainAndFFT_" + self.subject + '_electrode_' + self.electrode + '_'  '_tstamp_' + str(self.timestep) + 's_.png', dpi=250)


# example usage
# if __name__ == '__main__':
#
#     task1 = ComputeTask('testSubject','../DataSets/teData.mat',250,'O1')
#     task1.getDataSlice(2000,3000)
#     task1.SaveTimeFreqDomain()
#     task1.normDataSlice()
#     task1.computeRQA(td=2,emb=10,metric='Cosine')
#     task1.SaveRecurencePlot()