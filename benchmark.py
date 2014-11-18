#!/usr/bin/env python2
import itertools

import numpy as np
import matplotlib.pyplot as plt

from gazesim import JumpingObjectPositionSimulator as ObjectPositionSimulator, gaussian_noiser, generate_sequence
from saccade_detectors import ivt, idt, iocs as iocs_slow, mean_distance
#import pyximport; pyximport.install()
from fast_saccade_detectors import reconstruct_fixations, idt, optimize_1d, iocs

def gaze_mse(est, truth):
	diff = est - thruth

def benchmark():
	sampling_rate = 100.0
	duration = 10.0
	split_rate = 1.0/0.250
	
	stds = []
	ivt_errors = []
	idt_errors = []
	iocs_errors = []
	iocs_slow_errors = []
	measurement_errors = []
	minimum_errors = []
	for std in np.linspace(0.1, 10, 10):
		simulator = ObjectPositionSimulator(rate=split_rate)
		noiser = gaussian_noiser(sx=std, sy=std)
		generator = generate_sequence(simulator, noiser, sampling_rate=sampling_rate)
		data = zip(*itertools.islice(generator, int(duration*sampling_rate)))
		t, pos, gaze, saccades = map(np.array, data)
		saccades = np.flatnonzero(saccades)
		
		measurement_error = mean_distance(gaze, pos)
		minimum_error = mean_distance(reconstruct_fixations(gaze, saccades), pos)
		#plt.plot(t, zip(*pos)[0])
		#plt.plot(t, zip(*gaze)[0], '.')
		
		#optarg = optimize_1d(ivt, ((0, 10000),), t, gaze, pos)
		optarg = 1000.0
		ivt_saccades = ivt(t, gaze, optarg)
		ivt_gaze = reconstruct_fixations(gaze, ivt_saccades)
		ivt_error = mean_distance(ivt_gaze, pos)
		
		#optarg = optimize_1d(idt, ((0, 100),), t, gaze, pos)
		optarg = 20.0
		idt_saccades = idt(t, gaze, optarg)
		idt_gaze = reconstruct_fixations(gaze, idt_saccades)
		idt_error = mean_distance(idt_gaze, pos)
		idt_errors.append(idt_error)

		iocs_saccades = iocs(t, gaze, noise_std=std, split_rate=split_rate)
		iocs_gaze = reconstruct_fixations(gaze, iocs_saccades)
		iocs_error = mean_distance(iocs_gaze, pos)
		iocs_errors.append(iocs_error)
		
		iocs_slow_saccades = iocs_slow(t, gaze, noise_std=std, split_rate=split_rate)
		iocs_slow_gaze = reconstruct_fixations(gaze, iocs_slow_saccades)
		iocs_slow_error = mean_distance(iocs_slow_gaze, pos)
		iocs_slow_errors.append(iocs_slow_error)
		
		print len(saccades), len(iocs_saccades)
		ivt_errors.append(ivt_error)
		#measurement_errors.append(measurement_error)
		stds.append(std)
		minimum_errors.append(minimum_error)
	
	#plt.plot(stds, measurement_errors, color='black')
	plt.plot(stds, minimum_errors, color='black')
	plt.plot(stds, iocs_errors, 'o-', label='I-OCS')
	plt.plot(stds, iocs_slow_errors, 'o-', label='I-OCS-slow')
	#plt.plot(stds, ivt_errors, label='I-VT')
	plt.plot(stds, idt_errors, label='I-DT')
	plt.legend()
	plt.show()

def benchmark_perf():
	sampling_rate = 100.0
	duration = 1000.0
	split_rate = 1.0/0.250
	
	stds = []
	ivt_errors = []
	idt_errors = []
	iocs_errors = []
	measurement_errors = []
	minimum_errors = []
	
	std = 5.0
	simulator = ObjectPositionSimulator(rate=split_rate)
	noiser = gaussian_noiser(sx=std, sy=std)
	generator = generate_sequence(simulator, noiser, sampling_rate=sampling_rate)
	data = zip(*itertools.islice(generator, int(duration*sampling_rate)))
	ts, pos, gaze, saccades = map(np.array, data)
	saccades = np.flatnonzero(saccades)
	import time
	for i in range(100):
		t = time.time()
		iocs_saccades = iocs(ts, gaze, noise_std=std, split_rate=split_rate)
		new_t = time.time()
		print len(ts)/(new_t - t)
		t = new_t
		"""
		measurement_error = mean_distance(gaze, pos)
		minimum_error = mean_distance(reconstruct_fixations(gaze, saccades), pos)
		#plt.plot(t, zip(*pos)[0])
		#plt.plot(t, zip(*gaze)[0], '.')
		
		#optarg = optimize_1d(ivt, ((0, 10000),), t, gaze, pos)
		optarg = 1000.0
		ivt_saccades = ivt(t, gaze, optarg)
		ivt_gaze = reconstruct_fixations(gaze, ivt_saccades)
		ivt_error = mean_distance(ivt_gaze, pos)
		
		#optarg = optimize_1d(idt, ((0, 100),), t, gaze, pos)
		optarg = 20.0
		idt_saccades = idt(t, gaze, optarg)
		idt_gaze = reconstruct_fixations(gaze, idt_saccades)
		idt_error = mean_distance(idt_gaze, pos)
		idt_errors.append(idt_error)

		iocs_saccades = iocs(t, gaze, noise_std=std, split_rate=split_rate)
		iocs_gaze = reconstruct_fixations(gaze, iocs_saccades)
		iocs_error = mean_distance(iocs_gaze, pos)
		iocs_errors.append(iocs_error)
		
		print len(saccades), len(iocs_saccades)
		ivt_errors.append(ivt_error)
		#measurement_errors.append(measurement_error)
		stds.append(std)
		minimum_errors.append(minimum_error)
		"""


if __name__ == '__main__':
	benchmark()
