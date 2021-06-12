



def norm(arr, td=2, emb=9):
    tdemb = td * emb
    nslice = int(len(arr) / tdemb)
    array_size = (int(len(arr) / td), 1)
    temp = np.zeros(array_size)
    for i in range(nslice):

        j = i * tdemb
        nor = 0
        m = i * emb

        for k in range(emb):
            nor += (arr[j + k * td, 0]) * (arr[j + k * td, 0])

        nor = math.sqrt(nor)

        for l in range(emb):
            temp[int(m + l), 0] = arr[l * td + j, 0]  # /nor
    return temp

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
        for sample in range(1,samples_in_slice):
            xtemp[slice, sample] = arr[counter]
            ytemp[slice, sample] = arr[counter - 1]

            counter += 1

        # |X|  = sqrt(sum([x_i**2,x_1**2]))
        # |X|  = sqrt(Sum_i X_i ^ 2)
        normax = 1-math.sqrt(sum(xtemp[slice][i] ** 2 for i in range(samples_in_slice)))
        normay = 1-math.sqrt(sum(ytemp[slice][i] ** 2 for i in range(samples_in_slice)))
        # for i in range(samples_in_slice):
        #     norm_arr[slice,i] = (xtemp[slice][i] / normax) * (ytemp[slice][i] / normay)
        norm_arr[slice] = (xtemp[slice] / normax) * (ytemp[slice] / normay)
    return norm_arr.ravel()