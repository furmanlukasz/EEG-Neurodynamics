import numpy as np
from numpy import savetxt
import Neurodynamics
from argparse import ArgumentParser
from pyrqa.image_generator import ImageGenerator
from numpy import genfromtxt
import os
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

parser = ArgumentParser()
parser.add_argument('fromrange',type=int)
parser.add_argument('torange',type=int)
parser.add_argument('numChan',type=int)
parser.add_argument('d0',type=float)
parser.add_argument('d1',type=float)
args = parser.parse_args()

def linearMap(sig, d0, d1):
    d = d1-d0
    a = 1/d
    temp = np.zeros((sig.shape))
    for i in range(sig.shape[0]):
        if sig[i] < d0:
            temp[i] = 0
        elif sig[i] > d1:
            temp[i] = 1
        else:
            temp[i] = a*(sig[i]-d0)
    return temp


if __name__ == '__main__':
    print(("#" * 50) + ' PYRQA COMPUTATION STARTED ' + ("#" * 50))
    sock.sendto(msg_to_bytes(str(("#" * 10) + ' PYRQA COMPUTATION STARTED ' + ("#" * 10))), (upd_ip, udp_port))
    for i in range(args.numChan):

        MYDIR = ("Temp/pyrqa" + str(i))
        CHECK_FOLDER = os.path.isdir(MYDIR)

        if not CHECK_FOLDER:
            os.makedirs(MYDIR)
            print("created folder : ", MYDIR)

        sig = genfromtxt('Temp/sft' + str(i) + '/dist.csv', delimiter=',')

        sig = linearMap(sig,args.d0,args.d1)
        print(sig.shape)
        task = Neurodynamics.ComputeTask('testSubject',sig[args.fromrange:args.torange],250,0)
        result, result1 = task.computeRP(td=0,
                            emb=1,
                            metric='Cosine',
                            nbr=0.65)

        savetxt('Temp/pyrqa'+str(i)+'/rp_plot.csv', result.to_array(), delimiter=',')
        ImageGenerator.save_recurrence_plot(result1.recurrence_matrix_reverse,'Temp/pyrqa'+str(i)+'/rp_plot.jpg')
        print(("#" * 20) + ' RP plot saved ' + str(i) + ' ' +("#" * 20))
        progbar = progressBar(i, args.numChan)
        progbar = str(progbar).replace("[", '').replace("]", '').replace("'", '').replace(",", '')
        sock.sendto(msg_to_bytes(str(progbar)), (upd_ip, udp_port))

    print(("#" * 30) + ' PYRQA COMPUTATION FINISHED ' + ("#" * 30))
    sock.sendto(msg_to_bytes(str(("#" * 10) + ' PYRQA COMPUTATION FINISHED ' + ("#" * 10))), (upd_ip, udp_port))