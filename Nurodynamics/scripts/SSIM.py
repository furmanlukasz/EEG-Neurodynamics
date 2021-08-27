from argparse import ArgumentParser
import tensorflow as tf
parser = ArgumentParser()
parser.add_argument('eidx1',type=int)
parser.add_argument('eidx2',type=int)
args = parser.parse_args()
import socket
upd_ip = "127.0.0.1"
udp_port = 7001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def msg_to_bytes(msg):
    return msg.encode('utf-8')

if __name__ == '__main__':
    print(("#" * 50) + ' SSIM COMPUTATION STARTED ' + ("#" * 50))
    # Read images from file.
    im1 = tf.image.decode_image(tf.io.read_file('Temp/pyrqa' + str(args.eidx1) + '/rp_plot.png'))
    im2 = tf.image.decode_image(tf.io.read_file('Temp/pyrqa' + str(args.eidx2) + '/rp_plot.png'))
    tf.shape(im1)  # `img1.png` has 3 channels; shape is `(255, 255, 3)`
    print(tf.shape(im2))  # `img2.png` has 3 channels; shape is `(255, 255, 3)`
    # Add an outer batch for each image.
    im1 = tf.expand_dims(im1, axis=0)
    im2 = tf.expand_dims(im2, axis=0)

    # Compute SSIM over tf.uint8 Tensors.
    # ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
    #                       filter_sigma=1.5, k1=0.01, k2=0.03)

    # Compute SSIM over tf.float32 Tensors.
    im1 = tf.image.convert_image_dtype(im1, tf.float32)
    im2 = tf.image.convert_image_dtype(im2, tf.float32)
    print(tf.shape(im2))
    ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=2,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
    total_variation1 = tf.image.total_variation(im1)*0.0001
    total_variation2 = tf.image.total_variation(im2)*0.0001

    print('total variation 1: {} total variation 2: {} Diff: {}'.format(total_variation1,total_variation2,total_variation1-total_variation2))
    # ssim1 and ssim2 both have type tf.float32 and are almost equal.
    print(ssim2)
    print(("#" * 50) + ' SSIM COMPUTATION FINISHED ' + ("#" * 50))