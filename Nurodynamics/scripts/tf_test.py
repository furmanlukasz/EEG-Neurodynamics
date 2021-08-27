import os
from argparse import ArgumentParser
#start TouchDesigner.exe C:\Users\lukas\PycharmProjects\SystemDynamics\Nurodynamics\Nurodynamics.toe
import socket
from numpy import genfromtxt
from numpy import savetxt
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from scipy import signal
import matplotlib.pyplot as plt






#print(tensorflow.__version__)
upd_ip = "127.0.0.1"
udp_port = 7001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

parser = ArgumentParser()
parser.add_argument('fft_length',type=int)
parser.add_argument('frame_length',type=int)
parser.add_argument('frame_step',type=int)
parser.add_argument('eidx',type=int)
parser.add_argument('metric',type=str)
args = parser.parse_args()

sig = genfromtxt('Temp/sft'+str(args.eidx)+'/dataEEG.csv', delimiter=',')



#print(args.fft_length, args.frame_length, args.frame_step)
def msg_to_bytes(msg):
    return msg.encode('utf-8')
print('#'*80)
print('STFT start: fft_length={} frame_length={} frame_step={}'.format(str(args.fft_length), str(args.frame_length), str(args.frame_step)))
print('#'*80)
signal_tensor = tf.convert_to_tensor(sig, dtype=tf.float32)

stft_tensor = tf.signal.stft(signal_tensor, window_fn=tf.signal.hamming_window,fft_length=args.fft_length, frame_length=args.frame_length, frame_step=args.frame_step)
sft = np.array(stft_tensor)
sft = tf.abs(sft)
sft = sft[::,:50]

print('STFT computed')

print('distances start')
tempDist = np.zeros((sft.shape[0]))

for i in range(sft.shape[0]):
    if args.metric == 'cosine_dist':
        tempDist[i] = distance.cosine(sft[i],sft[i-1])
    elif args.metric == 'wasserstein_dist':
        tempDist[i] = wasserstein_distance(sft[i],sft[i-1])
    else:
        pass

savetxt('Temp/sft'+str(args.eidx)+'/sft.csv', sft, delimiter=',')
savetxt('Temp/sft'+str(args.eidx)+'/dist.csv', tempDist, delimiter=',')



print('#'*80)
print('STFT Settings: fft_length={} frame_length={} frame_step={}'.format(str(args.fft_length), str(args.frame_length), str(args.frame_step)))
print(sft.shape)
print('{} computed'.format(args.metric))
print('#'*80)
sock.sendto(msg_to_bytes(str('done')), (upd_ip, udp_port))
#plt.plot(tt)
#plt.pcolormesh(np.linspace(0,sig.shape[0]), np.linspace(0,10), sft, vmin=0, vmax=5, shading='gouraud')
#plt.show()
#print(callBack)