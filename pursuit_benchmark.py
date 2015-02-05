from __future__ import division

from functools import partial

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from gazesim import RandomLinearPursuitSimulator, gaussian_noiser, generate_sequence
from saccade_detectors import ivt, idt, iocs as iocs_slow, mean_distance, iols as iols_slow, reconstruct_pursuits
from fast_saccade_detectors import reconstruct_fixations, idt, optimize_1d, iocs, nols
from besseleye import eyefilter

sampling_rate = 100
duration = 10.0
dt = 1.0/sampling_rate
n = int(duration*sampling_rate)
split_rate = 1/0.250
nyquist_rate = 1.0/(2*dt)
# TODO: Refactor all the stuff to use mean duration
# instead of the rate, as the duration is much more
# intuitive.
sim = RandomLinearPursuitSimulator(mean_duration=1.0/split_rate)
#print [(dt*i, sim(dt)[0]) for i in range(1000)]
pos = np.array([sim(dt)[0] for i in range(n)])
#pos = np.zeros((250, 2))
t = np.arange(0, len(pos))*dt

#pos *= 0.0
#pos[:len(pos)/2.0][:,0] = t[:len(pos)/2.0]
#pos[:len(pos)/2.0][:,1] = t[:len(pos)/2.0]
#pos = np.zeros((11, 2))
#pos = np.zeros((100, 2))
#pos[len(pos)/2:] = 10.0

#pos = np.zeros((len(t), 2))
#pos[0] = 100

#print pos.shape
eye = eyefilter(pos, sampling_rate)
#eye = pos.copy()
#b, a = scipy.signal.bessel(2, hz_to_nyquist(15.0))
#eye = scipy.signal.lfilter(b, a, pos.T).T
#eye = np.vstack((np.sin(t*10), np.sin(t*10))).T

# TODO: Some weird weird bugs:
# One sample off:
#eye = np.vstack((t, t)).T*1000.0; eye[:len(eye)/2] = 10.0
# Totally batshit (due to pruning!)
#eye = np.vstack((t, t)).T*-100.0; eye[:len(eye)/2] = 10.0

# Reconstruction fails:
#eye = np.vstack((t, t)).T*-100.0; eye[0] = 10.0

# Not optimal!
#eye = np.vstack((t, t)).T*-100.0; eye[:4] = 10.0
#eye = np.zeros((len(t), 2))


# One sample off
#eye = np.vstack((t, t)).T*-100; eye[:len(eye)/2] = 10.0

#plt.plot(t, pos[:,0])

noise_std = 0.5
forced_split_rate = split_rate
iols_slow = partial(iols_slow, split_rate=None, noise_std=np.array([noise_std, noise_std]))
iocs = partial(iocs, split_rate=forced_split_rate, noise_std=np.array([noise_std, noise_std]))
nols = partial(nols, split_rate=forced_split_rate, noise_std=np.array([noise_std, noise_std]))
iocs_slow = partial(iocs_slow, noise_std=noise_std)
#iols_noprune = partial(iols_noprune, noise_std=np.array([noise_std, noise_std]))

def reconstruct(t, signal, func):
	splits = func(t, signal)
	return reconstruct_pursuits(t, signal, func(t, signal)), splits
	#return reconstruct_fixations(signal, func(t, signal))[:,0]
"""
errors = 0
niters = 100
for i in range(niters):
	signal = eye + np.random.randn(*eye.shape)*noise_std
	splits = iols_slow(t, signal)
	if splits != [0]:
		errors += 1

print errors, niters, errors/niters*100
errors = 0
for i in range(niters):
	signal = eye + np.random.randn(*eye.shape)*noise_std
	splits = iocs(t, signal)
	if len(splits) > 1:
		errors += 1
print errors, niters, errors/niters*100
"""

def optimize_noise_std(t, signal, func, noise_std):
	for i in range(100):
		print noise_std
		result, splits = reconstruct(t, signal, partial(func, noise_std=noise_std))
		noise_std = np.sqrt(np.sum((result-signal)**2/(len(result)-2), axis=0))
	return noise_std

signal = eye + np.random.randn(*eye.shape)*noise_std
#splits = iols_slow(t, signal)
#plt.subplot(2,1,1)
#plt.plot(signal[:,0], '.')
#plt.show()
plt.plot(t, eye[:,0])
plt.plot(t, signal[:,0], '.')

noise_std = noise_std*4.0
optimize_noise_std(t, signal, iols_slow, np.array([noise_std, noise_std]))

#newsig, splits = reconstruct(t, signal, iols_slow)
#plt.plot(t, newsig, label="IOLS")
#for i in splits:
#	plt.axvline(t[i])

#plt.plot(reconstruct(t, signal, iols_slow), label="IOLS")
#plt.plot(reconstruct(t, signal, nols), label="NOLS")
#plt.plot(reconstruct(t, signal, iocs), label="IOCS")
#plt.subplot(2,1,2)
#plt.plot(eye[:,1])
#plt.plot(signal[:,1], '.')
#plt.plot(t, reconstruct(t, signal, iols_noprune), '--', label="IOLS-noprune")
#plt.plot(t, reconstruct(t, signal, iocs_slow), '--', label="IOCS")
#plt.plot(signal[:,0], '.')

plt.legend()

plt.show()
