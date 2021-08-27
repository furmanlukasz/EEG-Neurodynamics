import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from numpy import genfromtxt
from mne.datasets import sample
from mne import setup_volume_source_space, setup_source_space
from mne import make_forward_solution
from mne.io import read_raw_fif
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('numLines',type=int)
args = parser.parse_args()
import socket
upd_ip = "127.0.0.1"
udp_port = 7001
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def msg_to_bytes(msg):
    return msg.encode('utf-8')

ssim = genfromtxt('Temp/SSIM_all/ssmi-all.csv', delimiter=',')

label_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
               'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
               'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3',
               'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7',
               'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7',
               'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7',
               'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
               'PO3', 'POz', 'PO4', 'PO8']
n_labels = len(label_names)
node_order = label_names
node_angles = circular_layout(label_names, start_pos=90, node_order=node_order,
                              group_boundaries=[0, len(label_names) / 2])
fig = plt.figure(num=None, figsize=(10, 10), facecolor='black')


plot_connectivity_circle(ssim, label_names, n_lines=args.numLines,
                         node_angles=node_angles,
                         title='SSIM Corelations',interactive=True, fig=fig)
fig.savefig('Temp/SSIM_all/graph.png', facecolor='black')
sock.sendto(msg_to_bytes(str("SSIM visualisation complete")), (upd_ip, udp_port))
#plt.show()