import numpy as np
import math
import Neurodynamics
import scipy
from numpy import asarray
from numpy import savetxt
'''To tylko miara odległości między sygnałami w czasie t i t', próbujemy jak najlepiej zwizualizowac informację widoczną na wykresach RR.
Bierzemy sygnał w kolejnych punktach czasu , X to wektor w chwili t, Y to wektor w chwili t-1, t-2, ... 1,
Wektory uwzględniają embedding, czyli reprezentują kawałek sygnału o długości Emb x td, np. 9x2=18 sampli. 
Odległość mapuje się na kolor(d(X,Y)).
Normalizacja wynika z tego, że
X×Y/|X||Y|=  X/|X| x Y/|Y| to iloczyn skalarny dwóch wektorów o długości 1. Jeśli są takie same to ten iloczyn = 1, czyli d(X,Y)=1-1=0
To mierzy podobieństwo sygnału w różnych punktach czasu. Liczymy na to, że będzie widać ciemne kwadraty koło diagonali tam, 
gdzie sygnał ma podobny kształt, a nie tylko małą amplitudę, a poza diagonalą będzie widać ciemne obszary pokazujące podobieństwo 
tego sygnału do przeszłego. Jeśli mamy w badanym oknie czasowym oscylacje niskiej częstotliwości o dużej amplitudzie to powinniśmy zobaczyć, 
czy w przeszłości były podobne, ale będą odległe od wysokiej częstotliwości z niską amplitudą.
Szukamy trochę na ślepo ale optymalizacja podobieństwa i kontrastu różnych sygnałów jest tu kluczowa i można będzie pomyśleć o algorytmie optymalizacji.
|X|=sqrt(XxX) = sqrt(Sum_i  X_i^2)



Norma wektora lub iloczynu to pierwiastek z sumy kwadratów jego elementów,
|X|=sqrt(XxX) = sqrt(Sum_i  X_i^2)
Dla iloczynu można najpierw wymnozyć |X|*|Y| i potem zrobić pierwiastek z iloczynu.

'''
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

    interval = 300
    td = 5
    emb = 20
    electrodeList = ['Fp2', 'F4', 'CP4', 'P4']
    electrodeList = ['CP4']

    task1 = Neurodynamics.ComputeTask('testSubject', data, 250, 'O1')
    #slice = task1.getDataSlice(2000,2900)
    # save numpy array as csv file


    # define data
    #data = slice
    # # save to csv file
    #savetxt('../data.csv', data, delimiter=',')
    slice = task1.testSignal(10, interval)
    task1.setDataSlice(slice)
    #task1.normDataSlice(npoints=interval, td=td, emb=emb)

    #print(norm(slice))
    # task1.normDataSlice()
    task1.SaveTimeFreqDomain()
    task1.SaveSpectrogram()
    #
    #
    task1.computeRP(td=td, emb=emb, metric='Cosine')
    task1.SaveRecurencePlot()
    #