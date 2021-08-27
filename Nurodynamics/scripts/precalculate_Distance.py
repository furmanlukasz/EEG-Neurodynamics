import os
from argparse import ArgumentParser
#start TouchDesigner.exe C:\Users\lukas\PycharmProjects\SystemDynamics\Nurodynamics\Nurodynamics.toe
import socket
from numpy import genfromtxt
from numpy import savetxt
import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from scipy import signal
import matplotlib.pyplot as plt






#print(tensorflow.__version__)
upd_ip = "127.0.0.1"
udp_port = 7001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

parser = ArgumentParser()
parser.add_argument('metric',type=str)
parser.add_argument('numChan',type=int)
args = parser.parse_args()

def progressBar(i,chanDim,text=""):
    chanDim -= i
    info = ['#']*chanDim
    return info


def msg_to_bytes(msg):
    return msg.encode('utf-8')

print('#'*50)
print('Distance start: metric={}'.format(args.metric))
sock.sendto(msg_to_bytes(str('Distance start: metric={}'.format(args.metric))), (upd_ip, udp_port))
print('#'*50)

if __name__ == '__main__':

    for i in range(args.numChan):
        sft = genfromtxt('Temp/sft' + str(i) + '/sft.csv', delimiter=',')

        tempDist = np.zeros((sft.shape[0]))

        for j in range(sft.shape[0]):
            if args.metric == 'cosine_dist':
                tempDist[j] = distance.cosine(sft[j],sft[j-1])
            elif args.metric == 'wasserstein_dist':
                tempDist[j] = wasserstein_distance(sft[j],sft[j-1])
            else:
                pass
        savetxt('Temp/sft' + str(i) + '/dist.csv', tempDist, delimiter=',')
        print(("#" * 20) + ' Dist saved ' + str(i) + ("#" * 20))
        progbar = progressBar(i, args.numChan)
        progbar = str(progbar).replace("[", '').replace("]", '').replace("'", '').replace(",", '')
        sock.sendto(msg_to_bytes(str(progbar)), (upd_ip, udp_port))
    print(("#" * 30) + ' DISTANCE: ALL COMPUTATION FINISHED ' + ("#" * 30))
    sock.sendto(msg_to_bytes(str(("#" * 10) + ' DISTANCE: ALL COMPUTATION FINISHED ' + ("#" * 10))), (upd_ip, udp_port))



