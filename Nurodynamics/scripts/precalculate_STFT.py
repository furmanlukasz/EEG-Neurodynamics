import os
from argparse import ArgumentParser
#start TouchDesigner.exe C:\Users\lukas\PycharmProjects\SystemDynamics\Nurodynamics\Nurodynamics.toe

from numpy import genfromtxt
from numpy import savetxt
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from scipy import signal
import matplotlib.pyplot as plt

import socket
upd_ip = "127.0.0.1"
udp_port = 7001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def msg_to_bytes(msg):
    return msg.encode('utf-8')

def progressBar(i,chanDim):
    chanDim -= i
    info = ['#']*chanDim
    return info
#print(tensorflow.__version__)


parser = ArgumentParser()
parser.add_argument('fft_length',type=int)
parser.add_argument('frame_length',type=int)
parser.add_argument('frame_step',type=int)
args = parser.parse_args()

sig = genfromtxt('Temp/FilteredData/filteredEEG.csv', delimiter=',')
print(sig.shape)

print('#'*80)
print('STFT start: fft_length={} frame_length={} frame_step={}'.format(str(args.fft_length), str(args.frame_length), str(args.frame_step)))
print('#'*80)
sock.sendto(msg_to_bytes(str('STFT start: fft_length={} frame_length={} frame_step={}'.format(str(args.fft_length), str(args.frame_length), str(args.frame_step)))), (upd_ip, udp_port))
##sock.sendto(msg_to_bytes(str()), (upd_ip, udp_port))
if __name__ == '__main__':

    for i in range(sig.shape[0]):
        signal_tensor = tf.convert_to_tensor(sig[i], dtype=tf.float32)
        stft_tensor = tf.signal.stft(signal_tensor, window_fn=tf.signal.hamming_window,fft_length=args.fft_length, frame_length=args.frame_length, frame_step=args.frame_step)
        sft = np.array(stft_tensor)
        sft = tf.abs(sft)
        sft = sft[::,:50]

        MYDIR = ('Temp/sft' + str(i))
        CHECK_FOLDER = os.path.isdir(MYDIR)
        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)

        savetxt('Temp/sft' + str(i) + '/sft.csv', sft, delimiter=',')
        print(("#" * 20) + ' STFT saved ' + str(i)+" " + ("#" * 20))
        progbar = progressBar(i, sig.shape[0])
        progbar = str(progbar).replace("[",'').replace("]",'').replace("'",'').replace(",",'')
        sock.sendto(msg_to_bytes(str(progbar)), (upd_ip, udp_port))
    print(("#" * 50) + ' STFT ALL COMPUTATION FINISHED ' + ("#" * 50))
    sock.sendto(msg_to_bytes(("#" * 10) + ' STFT ALL COMPUTATION FINISHED ' + ("#" * 10)), (upd_ip, udp_port))


    # tempDist = np.zeros((sft.shape[0]))
    #
    # for i in range(sft.shape[0]):
    #     if args.metric == 'cosine_dist':
    #         tempDist[i] = distance.cosine(sft[i],sft[i-1])
    #     elif args.metric == 'wasserstein_dist':
    #         tempDist[i] = wasserstein_distance(sft[i],sft[i-1])
    #     else:
    #         pass
    #
    # savetxt('Temp/sft'+str(args.eidx)+'/sft.csv', sft, delimiter=',')
    # savetxt('Temp/sft'+str(args.eidx)+'/dist.csv', tempDist, delimiter=',')
    #
    #
    #
    # print('#'*80)
    # print('STFT Settings: fft_length={} frame_length={} frame_step={}'.format(str(args.fft_length), str(args.frame_length), str(args.frame_step)))
    # print(sft.shape)
    # print('{} computed'.format(args.metric))
    # print('#'*80)
    # sock.sendto(msg_to_bytes(str('done')), (upd_ip, udp_port))
