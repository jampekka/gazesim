import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def stepsignal(duration, rate):
	#hittime = duration*step_at_duration
	t = np.arange(0.0, duration, 1.0/rate)
	signal = np.ones(len(t))
	signal[0] = 0.0
	return t, signal

def hz_to_nyquist(hz, sampling_rate):
	nyquist_rate = sampling_rate/2.0
	return hz/sampling_rate

# Default parameters estimated from empirical fit.
def eyefilter(signal, sampling_rate, order=3, cutoff=33.0):
	# TODO: Use analog version and/or better digitalization
	# TODO: The fit doesn't seem stable, may not be a global
	# error minimum
	b, a = scipy.signal.bessel(order, hz_to_nyquist(cutoff, sampling_rate))
	return scipy.signal.lfilter(b, a, signal)

# From Table 1 of "Variability and development of a normative
# data base for saccadic eye movements", Bahill et al 1989.
SACCADE_DATA = np.rec.fromarrays([
	[5, 10, 15, 20],
	[261, 410, 499, 657],
	np.array([42, 51, 54, 64])/1000.0],
	names='magnitudes,peak_velocities,durations'
	)
def empiric_saccade_fit(d=SACCADE_DATA):
	# NOTE: Seems to be approximatedly linear only
	# in around the range 5 - 20 deg saccades. Should work
	# for our purposes though. It's difficult
	# to say from the article whether individual subjects
	# have the same "nonlinearity" than the aggregate data.
	# especially with very small saccades, as the noise seems
	# to be quite heteroschedastic (see Fig 3 of ibid).
	mag_to_vel = np.poly1d(np.polyfit(d.magnitudes, d.peak_velocities, 1))
	mag_to_dur = np.poly1d(np.polyfit(d.magnitudes, d.durations, 1))
	
	#rng = magnitudes
	#plt.scatter(magnitudes, peak_velocities)
	#plt.plot(magnitudes, mag_to_vel(np.array(magnitudes)))
	
	#plt.scatter(magnitudes, durations)
	#plt.plot(rng, mag_to_dur(np.array(rng)))
	#plt.show()
	return (mag_to_vel, np.std(d.peak_velocities)), (mag_to_dur, np.std(d.durations))

def saccade_features(t, result, mag, minmag=0.0):
	if mag < minmag:
		mag = minmag
	velocity = np.diff(result)/np.diff(t)
	try:
		dur = t[np.flatnonzero(result >= mag-minmag)[0]]
	except IndexError:
		dur = t[-1]
	
	return np.max(velocity), dur

def estimate_eyefilter_rate(sampling_rate, guess=10.0, order=3):
	# TODO: This could probably be solved analytically
	probe_magnitudes = np.array([5, 10, 15, 20])
	#probe_magnitudes = np.array([5])
	probe_duration = 3.0
	t, step = stepsignal(probe_duration, sampling_rate)
	dt = 1.0/sampling_rate
	probe_signals = [step*m for m in probe_magnitudes]
	(mag_to_vel, mag_to_vel_scaler), (mag_to_dur, mag_to_dur_scaler) = empiric_saccade_fit()

	def evaluate_args((cutoff,)):
		feats = []
		#print cutoff
		if cutoff < 0:
			print "ARGH, UNDER BOUNDS!"
			cutoff = 0.0
		if cutoff > rate/2.0:
			print "ARGH, OVER BOUNDS!"
			cutoff = rate/2.0
		for mag, sig in zip(probe_magnitudes, probe_signals):
			result = eyefilter(sig, sampling_rate, cutoff=cutoff, order=order)
			feats.append(saccade_features(t, result, mag))
		
			#plt.subplot(2,1,1)
			#plt.plot(result)
		
		empiric_maxs, empiric_durations = zip(*feats)
		#plt.subplot(2,1,2)
		#plt.suptitle(cutoff)
		#plt.plot(probe_magnitudes, empiric_maxs)
		#plt.plot(probe_magnitudes, empiric_durations)
		#plt.show()
		
		velerr = np.subtract(empiric_maxs, mag_to_vel(probe_magnitudes))**2/mag_to_vel_scaler**2
		durerr = np.subtract(empiric_durations, mag_to_dur(probe_magnitudes))**2/mag_to_dur_scaler**2

		#print velerr, durerr, mag_to_dur_scaler, empiric_durations
		# TODO: Velocity and duration aren't necessarily commesurable
		return np.mean((velerr+durerr))#+durerr))
	
	result = scipy.optimize.minimize(evaluate_args, [guess], bounds=[(0, rate/2.0)], options=dict(disp=True))
	#rng = np.linspace(5, 50, 1000)
	#plt.plot(rng, [evaluate_args([v]) for v in rng])
	#plt.show()
	return float(result.x), result.fun

def estimate_eyefilter_parameters(rate):
	orders = range(1, 11)
	errs = []

	for order in orders:
		result = estimate_eyefilter_rate(rate, order=order)
		errs.append(result)
	
	cutoffs, errors = zip(*errs)
	plt.plot(orders, errors)
	winner = np.argmin(errors)
	cutoff = cutoffs[winner]
	order = orders[winner]
	print "Fitted params eyefilter params: %f, %i"%(cutoff, order)
	return lambda sig, cutoff=cutoff, order=order: eyefilter(sig, rate, cutoff=cutoff, order=order)
	

if __name__ == '__main__':
	rate = 10.0
	duration = 3.0
	step_10hz = stepsignal(duration, rate)
	#plt.plot(step_10hz[0], step_10hz[1])
	#plt.plot(step_10hz[0], eyefilter(step_10hz[1], rate))
	
	rate = 1000.0
	filt = estimate_eyefilter_parameters(rate)
	
	step_1000hz = stepsignal(duration, rate)
	
	testmags = np.linspace(0.0, SACCADE_DATA.magnitudes.max(), 200)
	results = []
	for mag in testmags:
		results.append(saccade_features(
			step_1000hz[0],
			filt(step_1000hz[1]*mag), mag)
			)
	
	vels, durs = zip(*results)
	plt.subplot(2,1,1)
	plt.scatter(SACCADE_DATA.magnitudes, SACCADE_DATA.peak_velocities)
	plt.plot(testmags, vels)
	plt.subplot(2,1,2)
	plt.scatter(SACCADE_DATA.magnitudes, SACCADE_DATA.durations)
	plt.plot(testmags, durs)


	#plt.plot(step_1000hz[0], step_1000hz[1])
	#plt.plot(step_1000hz[0], filt(step_1000hz[1]))
	#plt.plot(step_1000hz[0], eyefilter(step_1000hz[1], rate, order=9))
	plt.show()
