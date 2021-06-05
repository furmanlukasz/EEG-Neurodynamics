from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import scipy.fftpack
import h5py
from scipy.io import loadmat
from scipy.signal import welch
from nolitsa import dimension
import pywt
from pyrqa.image_generator import ImageGenerator
from pyrqa.computation import RPComputation
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import FixedRadius, RadiusCorridor
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import Unthresholded

from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator

def loadData(filename):
    data = loadmat(filename)
    electrodes = data['chanlocs']['labels']
    return np.array(data['EEGdata']),[electrodes[0][i][0] for i in range(len(electrodes[0]))]

def setDataSlice(df,fromrange, torange):
    ndata = np.zeros((len(df), torange-fromrange))
    for i in range(len(df)):
        ndata[i] = df[i][fromrange:torange]
    return ndata


df, electrodeName = loadData('../DataSets/teData.mat')
dane = setDataSlice(df, 2000, 6000)
bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high gamma']
selected_band = 3
time = np.arange(len(dane[0]))
npnts = len(time)
db4 = pywt.Wavelet('db4')
coeffs = pywt.wavedec(dane[0], db4, mode='periodic', level=5)
coeffs2 = pywt.wavedec(dane[1], db4, mode='periodic', level=5)
f = signal.resample(coeffs[selected_band], npnts)
f2 = signal.resample(coeffs2[selected_band], npnts)

data_points_x = dane[8]
time_series_x = TimeSeries(data_points_x,
                           embedding_dimension=5,
                           time_delay=1)
data_points_y = dane[22]
time_series_y = TimeSeries(data_points_y,
                           embedding_dimension=5,
                           time_delay=1)
time_series = (time_series_x,
               time_series_y)
settings = Settings(time_series,
                    analysis_type=Cross,
                    #neighbourhood=FixedRadius(0.73),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=0)
computation = RQAComputation.create(settings,
                                    verbose=True)
result = computation.run()
result.min_diagonal_line_length = 2
result.min_vertical_line_length = 2
result.min_white_vertical_line_length = 2
print(result)
computation = RPComputation.create(settings)
result = computation.run()
fig = plt.figure(figsize=(8, 8))
grid = plt.GridSpec(6, 6, hspace=0.6, wspace=0.6)
main_ax = fig.add_subplot(grid[:-1, 1:])

x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
x_hist2 = fig.add_subplot(grid[1, 1:], yticklabels=[], sharex=x_hist)
main = main_ax.imshow(result.recurrence_matrix_reverse, interpolation='none',origin='upper')
x_hist.plot(data_points_x, color='gray')
x_hist2.plot(data_points_y, color='gray')
cbar_ax = fig.add_axes([0.04, 0.10, 0.05, 0.7])
fig.colorbar(main, cax=cbar_ax)
main_ax.invert_yaxis()

plt.savefig(
    "../RR_plots/Cross2_.png", dpi=500)

ImageGenerator.save_recurrence_plot(result.recurrence_matrix_reverse,
                                    'cross_recurrence_plot.png')