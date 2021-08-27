from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from numpy import savetxt
import timeit
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
parser.add_argument('numChan',type=int)
parser.add_argument('maxVal',type=float)
parser.add_argument('filterSize',type=float)
parser.add_argument('filterSigma',type=float)
parser.add_argument('k1',type=float)
parser.add_argument('k2',type=float)
args = parser.parse_args()

temp = np.zeros((args.numChan,args.numChan))
if __name__ == '__main__':
    start_tensor = timeit.default_timer()
    print(("#" * 50) + ' SSIM ALL COMPUTATION STARTED ' + ("#" * 50))
    sock.sendto(msg_to_bytes(str(("#" * 10) + ' SSIM ALL COMPUTATION STARTED ' + ("#" * 10))), (upd_ip, udp_port))
    k = 0
    for k in range(args.numChan):
        for i in range(args.numChan):

            im1 = tf.image.decode_image(tf.io.read_file('Temp/pyrqa' + str(k) + '/rp_plot.jpg'))
            im2 = tf.image.decode_image(tf.io.read_file('Temp/pyrqa' + str(i) + '/rp_plot.jpg'))
            #print(tf.shape(im1))
            #tf.shape(im2)
            # Add an outer batch for each image.
            im1 = tf.expand_dims(im1, axis=0)
            im2 = tf.expand_dims(im2, axis=0)
            # Compute SSIM over tf.float32 Tensors.
            im1 = tf.image.convert_image_dtype(im1, tf.float32)
            im2 = tf.image.convert_image_dtype(im2, tf.float32)
            #print(tf.shape(im2))
            ssim1 = tf.image.ssim(im1, im2, max_val=1, filter_size=2,
                               filter_sigma=1.5, k1=0.01, k2=0.03)
            #ssim2 = tf.image.ssim(im1, im2, max_val=args.maxVal, filter_size=int(args.filterSize),
            #                      filter_sigma=args.filterSigma, k1=args.k1, k2=args.k2)
            temp[k][i] = ssim1

            print(("#" * 20) + ' SSIM saved i={} k={} ssim={} '.format(i, k,ssim1) + ' ' + ("#" * 20))
            progbar = progressBar(i, args.numChan)
            progbar = str(progbar).replace("[", '').replace("]", '').replace("'", '').replace(",", '')
            sock.sendto(msg_to_bytes(str(("#" * 10) + ' SSIM saved i={} k={} ssim={} '.format(i, k,ssim1) + ' ' + ("#" * 10))), (upd_ip, udp_port))
            sock.sendto(msg_to_bytes(str(progbar)),(upd_ip, udp_port))
            if i == args.numChan:
                k += 1
            else:
                k += 0

    savetxt('Temp/SSIM_all/ssmi-all.csv', temp, delimiter=',')
    runtime_tensor = timeit.default_timer() - start_tensor
    print(runtime_tensor)
    print(("#" * 50) + ' SSIM ALL COMPUTATION FINISHED ' + ("#" * 50))
    sock.sendto(msg_to_bytes(str(("#" * 10) + ' SSIM ALL COMPUTATION FINISHED ' + ("#" * 10))), (upd_ip, udp_port))
    print(temp)
    # total_variation1 = tf.image.total_variation(im1)*0.0001
    # total_variation2 = tf.image.total_variation(im2)*0.0001
    #
    # print('total variation 1: {} total variation 2: {} Diff: {}'.format(total_variation1,total_variation2,total_variation1-total_variation2))
    # # ssim1 and ssim2 both have type tf.float32 and are almost equal.
    # print(ssim2)
