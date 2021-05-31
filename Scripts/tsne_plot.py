import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.manifold import TSNE

from pyrqa.image_generator import ImageGenerator
from pyrqa.computation import RPComputation
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius, RadiusCorridor
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import Unthresholded
# with open('j.pickle', 'wb') as handle:
#     pickle.dump(j, handle, protocol=pickle.HIGHEST_PROTOCOL)


# j - el. C6, beta, 8-12 sekunda
with open('j.pickle', 'rb') as handle:
    j = pickle.load(handle)

elnames=['C6']
el_idx=0
nazwy=['delta','theta','alpha','beta','gamma','high gamma']
a=3
timedel=25
embedding=3
interval=1000
counter=3

##RR
time_series = TimeSeries(j,
                        embedding_dimension=embedding,
                        time_delay=timedel)
settings = Settings(time_series,
                    analysis_type=Classic,
                    neighbourhood=Unthresholded(),
                    #neighbourhood=RadiusCorridor(inner_radius=0.32,outer_radius=0.86),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=1)
computation = RPComputation.create(settings,
                                verbose=True)
result = computation.run()

result.min_diagonal_line_length = 2
result.min_vertical_line_length = 2
result.min_white_vertical_line_length = 2

plt.imshow(result.recurrence_matrix_reverse_normalized, cmap='jet', interpolation='none')
plt.colorbar()
plt.title(elnames[el_idx]+', '+ nazwy[a] +' band, ' + 'emb = '+ str(embedding) +' td = '+ str(timedel) + ' timestamp ' + str((interval*counter)/250))
plt.show()

###TSNE
start=0
count=interval-(embedding-1)*timedel


recurrence_vectors = []

for dim in np.arange(embedding):
    offset = dim * timedel
    recurrence_vectors.append(j[(start + offset):(start + offset + count)])

vector2=np.array(recurrence_vectors, dtype='object').transpose()

vector=np.zeros(((len(vector2)+(embedding-1)*timedel), embedding))


for dim in np.arange(embedding):
    vector[dim*timedel:len(vector2)+dim*timedel, dim]=vector2[:,dim]
    # print(vector)

df_vector=pd.DataFrame(vector)
df_vector.index.name="time"

"""
Kolumny odpowiadają wartością przesuniętym o td w zależności od wymiaru zanurzenia. 
Wartości w klumnach zostały przesunięte tak, aby dana wartość w każdej z nich pasowała do czasu - indexu time
Wektory zanurzenia są w rzędach, w tym wypadku emb = 3
"""
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=5000)
tsne_results = tsne.fit_transform(df_vector.values)


df_vector['tsne-2d-one'] = tsne_results[:,0]
df_vector['tsne-2d-two'] = tsne_results[:,1]

"""
plotowane są komponenty tsne dla rzędów, w których wszystkie wartości są niezerowe
"""
plt.scatter(df_vector['tsne-2d-one'].iloc[(embedding-1)*timedel:len(vector)-((embedding-1)*timedel)],df_vector['tsne-2d-two'].iloc[(embedding-1)*timedel:len(vector)-((embedding-1)*timedel)],c=df_vector.index.values[(embedding-1)*timedel:len(vector)-((embedding-1)*timedel)],cmap='Spectral', alpha=0.4)
plt.colorbar()
plt.xlabel('tsne-2d-one')
plt.ylabel('tsne-2d-two')

plt.title(elnames[el_idx]+', '+ nazwy[a] +' band, ' + 'emb = '+ str(embedding) +' td = '+ str(timedel) + ' timestamp ' + str((interval*counter)/250))
plt.show()