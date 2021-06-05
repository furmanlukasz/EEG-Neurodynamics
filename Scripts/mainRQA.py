import Neurodynamics

if __name__ == '__main__':


    data = Neurodynamics.loadData('../DataSets/teData.mat')
    electrodeList = data[1]

    task1 = Neurodynamics.ComputeTask('testSubject',data,250,electrodeList[10])
    task1.getDataSlice(2000,3000)
    task1.SaveTimeFreqDomain()
    task1.normDataSlice()
    task1.computeRQA(td=2,emb=10,metric='Cosine')
    task1.SaveRecurencePlot()