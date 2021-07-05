import numpy as np
import math
import Neurodynamics
import scipy
from numpy import asarray
from numpy import savetxt

def norm(arr, td=2, emb=9):
    # definicja zmiennych pomocniczych
    samples_in_slice = td * emb
    num_slices = int(len(arr) / samples_in_slice)
    array_size = (num_slices, samples_in_slice)

    # pre-definicja szeregów
    norm_arr = np.zeros(array_size)

    xtemp = np.zeros(array_size)
    ytemp = np.zeros(array_size)

    counter = 0
    for slice in range(num_slices):
        for sample in range(samples_in_slice):
            xtemp[slice, sample] = arr[counter]
            ytemp[slice, sample] = arr[counter - 1]

            counter += 1

        # |X|  = sqrt(sum([x_i**2,x_1**2]))
        # |X|  = sqrt(Sum_i X_i ^ 2)
        normax = math.sqrt(sum(xtemp[slice][i] ** 2 for i in range(samples_in_slice)))
        normay = math.sqrt(sum(ytemp[slice][i] ** 2 for i in range(samples_in_slice)))

        norm_arr[slice] = (xtemp[slice] / normax) * (ytemp[slice] / normay)

    return norm_arr.ravel()





if __name__ == '__main__':

    data = Neurodynamics.loadData('../DataSets/teData.mat')
    electrodeList = data[1]

    interval = 900
    td = 2
    emb = 9
    electrodeList = ['Fp2', 'F4', 'CP4', 'P4']
    electrodeList = ['CP4']

    task1 = Neurodynamics.ComputeTask('testSubject', data, 250, 'O1')
    slice = task1.getDataSlice(0,10240)
    # # save numpy array as csv file
    #
    #
    # # define data
    # data = slice
    # # save to csv file
    # savetxt('../dataName.csv', data[1], delimiter=',', fmt='%s')
    #slice = task1.testSignal(10, interval)
    task1.setDataSlice(slice)
    task1.normDataSlice(npoints=interval, td=td, emb=emb)

    #print(norm(slice))
    # task1.normDataSlice()
    #task1.SaveTimeFreqDomain()
    #task1.SaveSpectrogram()
    task1.SFT()

    #
    #
    #task1.computeRP(td=td, emb=emb, metric='Cosine')
    #task1.SaveRecurencePlot('savedData_norm')
    #