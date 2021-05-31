
# Recurrence quantification analysis of EEG signals.

### W. Duch, K. Tołpa, S. Duda, Ł. Furman, M. Lewandowska. J. Dreszer.
### Neurcognitive Laboratory, Center for Modern Interdisciplinary Technologies, Nicolaus Copernicus University,

Spotting fingerprints of mental activity in brain EEG signals is quite difficult. EEG microstates are very short meta-stable distributions of electrical potentials over the whole scalp. Cross-recurrence plots may show more precisely patterns of meta-stable states that result from synchronization of oscilations between different parts of the brain. We are using various types of hi-density (128 channels) EEG recordings, including resting state and Multi-Source Interference Task (MSIT) experiments. Raw signals have been cleaned from artifacts, leaving 10.000 samples in 40 second segments. These signals are filtered using wavelet analysis into 5 narrow frequency bands, delta, theta, alpha, beta and gamma. In each case we have searched for combination of embedding, time-delay and similarity measures that can be used to expose similar states, first for single channels and next for all channels.

EEG signals are very difficult to analyze, and standard methods of parameter estimation, such as the false nearest neighborhoods (FNNs) for the embedding dimension m, and delayed mutual information for choosing the delay time td, may not be sufficient. Metastable states are characterized by a long trapping times (TT). For fixed embedding dimension both FNN and TT are rather stable and there is no correlation between their values. For low embedding dimensions and small time delays discrete square structures are formed along the diagonal, indicating that noisy point attractor states are present, separated by quick transitions due to high amplitude random oscillations. Non-linear mapping of distances to colors help to smooth the plots and increase contrast. For neural signals recurrence threshold may be replaced with soft version based on logit function simulating neural excitability. It should have relatively small values for distances below some threshold, showing high similarity between states that are fluctuations around average vector inside the attractor basins. For larger differences between state vectors logit values will quickly grow, showing significant differences.

The hope is that proper selection of parameters and similarity measures between different segments of the signal will help to reveal dynamics of activation of various large-scale brain subnetworks.

References

J. Dreszer, M. Grochowski, M. Lewandowska, J. Nikadon, J. Gorgol, B. Bałaj, K. Finc, W. Duch, T. Piotrowski. Spatiotemporal Complexity Patterns of the Frontoparietal Network Predict Fluid Intelligence: Sex Matters. Human Brain Mapping 41(17), 4846-4865, 2020.

## (WIP)
![](icon.png)
![](RR_plots/Dist_IB2018A0Z63922_rso_C5_beta_emb_2_td_8_tstamp_16.0.png)
![](RR_plots/Dist_IB2018A0Z63922_rso_C6_beta_emb_2_td_8_tstamp_16.0.png)
![](RR_plots/Dist_IB2018A0Z63922_rso_Cz_beta_emb_2_td_8_tstamp_16.0.png)
![](RR_plots/Dist_IB2018A0Z63922_rso_Fz_beta_emb_2_td_8_tstamp_16.0.png)

