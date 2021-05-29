
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.fftpack


from pyrqa.image_generator import ImageGenerator
from pyrqa.computation import RPComputation
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius, RadiusCorridor
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import Unthresholded

from nolitsa import dimension
import pywt
import pickle



def plotSignal(data):
  fig,ax = plt.subplots(1,2,figsize=(20,6))
  ax[0].plot(time,data)
  ax[0].set_xlabel('Time (s)')
  ax[0].set_title('Time domain')

  dataX = scipy.fftpack.fft(data/npnts)
  ax[1].plot(hz,np.abs(dataX[:len(hz)]))
  ax[1].set_xlim([0,150])
  ax[1].set_xlabel('Frequency (Hz)')
  ax[1].set_title('Frequency domain')




if __name__ == '__main__':

    filename = 'dane.pkl'
    df=pickle.load(open(filename,'rb'))
    srate = 256
    len(df[0])
    time = np.arange(len(df[0]))
    npnts = len(time)
    hz = np.linspace(0, srate / 2, int(npnts / 2) + 1)
    plotSignal(np.array(df[0]))
    plt.show()
    dane = df #FZ
    #matdata = loadmat('sampleEEGdata.mat')
    #matdata.items()
    elnames = ['Fz']  # C5, C6, Cz, Fz

    timedel = 8

    embedding = 2  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

    interval = 1000

    fnn_list = []
    tt_list = []
    fnn_dic = {}
    tt_dic = {}
    timestamps = []
    el_idx = 0
    subject = 'IB2018A0Z63922_rso'


    db4 = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(dane[el_idx], db4, mode='periodic', level=5)
    dlu = len(dane[el_idx])
    nazwy = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high gamma']

    a = 3

    f = signal.resample(coeffs[a], dlu)
    if interval:
        list_of_slices = zip(*(iter(f),) * interval)
        counter = 1
    for j in list_of_slices:

        fnn = np.asscalar(dimension.fnn(j, dim=[embedding], tau=timedel, metric='euclidean')[2])


        time_series = TimeSeries(j,
                                 embedding_dimension=embedding,
                                 time_delay=timedel)
        settings = Settings(time_series,
                            analysis_type=Classic,
                            neighbourhood=Unthresholded(),
                            # neighbourhood=RadiusCorridor(inner_radius=0.32,outer_radius=0.86),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1)
        computation = RPComputation.create(settings,
                                           verbose=True)
        result = computation.run()


        plt.imshow(result.recurrence_matrix_reverse_normalized[::-1], cmap='jet', interpolation='none', origin='upper')
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title(elnames[el_idx] + ', ' + nazwy[a] + ' band, ' + 'emb = ' + str(embedding) + ' td = ' + str(
            timedel) + ' timestamp ' + str((interval * counter) / 250))
        plt.savefig(
             "Dist_" + subject + '_' + elnames[el_idx] + '_' + nazwy[a] + '_embedding_' + str(embedding) + '_td_' + str(
                 timedel) + '_timestamp_' + str((interval * counter) / 250) + '_v4s' + '.png',  dpi=500)
        plt.close()
        ###################################
        #ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse_normalized,'fz_tests.png')
        settings = Settings(time_series,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(0.65),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1)

        computation = RQAComputation.create(settings,
                                            verbose=True)
        result = computation.run()
        result.min_diagonal_line_length = 2
        result.min_vertical_line_length = 2
        result.min_white_vertical_line_length = 2

        rqaArray = result.to_array()
        tt = rqaArray[10]
        fnn_dic[str((interval * counter) / 250)] = fnn
        tt_dic[str((interval * counter) / 250)] = tt
        timestamps.append((interval * counter) / 250)
        fnn_list.append(fnn)
        tt_list.append(tt)
        counter += 1

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.scatter(timestamps, fnn_list, color="red", alpha=0.3)
    ax2.scatter(timestamps, tt_list, color="blue", alpha=0.3)
    ax.set_ylabel("FNN", color="red")
    ax2.set_ylabel("TT", color="blue", rotation=270)

    ax.set_yticks(np.round(np.linspace(np.min(fnn_list), np.max(fnn_list), 10), 2))
    ax2.set_yticks(np.round(np.linspace(np.min(tt_list), np.max(tt_list), 10), 2))
    plt.xticks(timestamps)
    ax.set_xlabel('Timestamps')
    plt.title("Electrode C6 " + "beta" + " band")

    plt.savefig("FNN_TT" + "_C6_beta_" + '.png')
    plt.close()

    plt.scatter(tt_list, fnn_list)
    plt.xlabel("Trapping times")
    plt.ylabel("Fnns")
    plt.show()


