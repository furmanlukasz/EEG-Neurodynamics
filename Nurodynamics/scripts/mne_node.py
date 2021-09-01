from mne.externals.pymatreader import read_mat
import mne
import numpy as np
import os.path as op
from mayavi import mlab
from mne.datasets import fetch_fsaverage
# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)


fname = "../DataSets/teData.mat"
trans = 'fsaverage'
mat_data = read_mat(fname)
print(mat_data.keys())
print(mat_data['chanlocs']['labels'])

info = mne.create_info(mat_data['chanlocs']['labels'], 250., 'eeg')
raw = mne.io.RawArray(mat_data['EEGdata'], info)
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# Clean channel names to be able to use a standard 1005 montage
new_names = dict(
    (ch_name,
     ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
    for ch_name in raw.ch_names)
raw.rename_channels(new_names)

# Read and set the EEG electrode locations
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)
raw.set_eeg_reference(projection=True)  # needed for inverse modeling

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    raw.info, src=src, eeg=['original', 'projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials')

mlab.show()

fwd = mne.make_forward_solution(raw.info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)
print(fwd)

# Use fwd to compute the sensitivity map for illustration purposes
eeg_map = mne.sensitivity_map(fwd, ch_type='eeg', mode='fixed')
brain = eeg_map.plot(time_label='EEG sensitivity', subjects_dir=subjects_dir,
                     clim=dict(lims=[5, 50, 100]))
mlab.show()
#info = mne.create_info(mat_data)