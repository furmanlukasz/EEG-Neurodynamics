import Neurodynamics

if __name__ == '__main__':

    data = Neurodynamics.loadData('../DataSets/teData.mat')
    electrodeList = data[1]

    interval = 500
    electrodeList = ['Fp2', 'F4', 'CP4', 'P4']
    electrodeList = ['CP4']
    td = 2
    emb = 9
    for electrode in electrodeList:

        task = Neurodynamics.ComputeTask('testSubject', data, 250, 'CP4')
        chanLen = task.getChanLenght()

        EEGsamples = task.getDataSlice(0,chanLen)
        list_of_slices = zip(*(iter(EEGsamples),) * interval)
        counter = 1

        for slice in list_of_slices:
            task.setDataSlice(slice,counter)
            #task.normDataSlice(npoints=interval, td=td, emb=emb)
            task.SaveFreqDomain()
            task.SaveSpectrogram()

            task.computeRP(td=td, emb=emb, metric='Cosine')
            task.SaveRecurencePlot(title='RAW')
            counter +=1








    # task1 = Neurodynamics.ComputeTask('testSubject', data, 250, 'CP4')
    # task1.getDataSlice(2000,3000)
    # task1.normDataSlice()
    # task1.SaveTimeFreqDomain()
    # task1.SaveSpectrogram()
    #
    #
    # task1.computeRP(td=2,emb=9,metric='Cosine')
    # task1.SaveRecurencePlot()
    #














