import numpy as np
import scipy.optimize, scipy.stats
import operator

def norm(a, b):
	return np.sqrt(np.sum((a - b)**2, axis=1))

def mean_distance(a, b):
	return norm(a, b).mean()

def erode_consecutive(idx):
	return idx[:-1][np.diff(idx) > 1]

def ivt(t, gaze, threshold=1000.0):
	dt = np.diff(t)
	gaze = np.array(gaze)
	dists = np.sqrt(np.sum(np.diff(gaze, axis=0)**2, axis=1))
	speeds = dists/dt

	saccades = np.flatnonzero(speeds > threshold) + 1

	return erode_consecutive(saccades)

def idt(t, gaze, threshold=10.0):
	xmin = ymin = np.inf
	xmax = ymax = -np.inf
	saccades = []
	for i, (x, y) in enumerate(gaze):
		xmin = min(x, xmin)
		xmax = max(x, xmax)
		ymin = min(y, ymin)
		ymax = max(y, ymax)
		dispersion = (xmax - xmin) + (ymax - ymin)
		if dispersion > threshold:
			saccades.append(i)
			xmin = ymin = np.inf
			xmax = ymax = -np.inf
	return np.array(saccades)

class SplitHypothesis(object):
	@property
	def total_lik(self):
		return self.history_lik + self.segment_lik

def iocs(ts, gaze, rate=1.0/0.250, noise_std=1.0):
	seg_normer = np.log(1.0/(noise_std**2*np.pi*2))
	
	root_hypothesis = SplitHypothesis()
	root_hypothesis.splits = []
	root_hypothesis.history_lik = 0.0
	root_hypothesis.segment_lik = seg_normer
	root_hypothesis.n = 1
	root_hypothesis.mx = gaze[0][0]
	root_hypothesis.my = gaze[0][1]
	root_hypothesis.ssx = 0.0
	root_hypothesis.ssy = 0.0

	lik_comparator = lambda hypo: -hypo.total_lik 
	# TODO: Verify this! There's something wrong
	# either here or in the regression likelihood.
	# The split likelihood seems to be too large.
	split_lik = lambda dt: 2*np.log(1 - np.exp(-rate*dt))
	#split_lik = lambda dt: -20.0 # Works quite nicely with this. :'(
	
	prev_t = ts[0]
	hypotheses = [root_hypothesis]
	for i, (t, (x, y)) in enumerate(zip(ts, gaze)[1:], 1):
		dt = t - prev_t
		prev_t = t

		winner = hypotheses[0]
		new = SplitHypothesis()
		my_split_lik = split_lik(dt)
		new.history_lik = winner.total_lik + my_split_lik
		new.segment_lik = seg_normer
		new.n = 1
		new.splits = winner.splits + [i]
		new.mx = x
		new.my = y
		new.ssx = 0.0
		new.ssy = 0.0
		
		# The hypotheses that have their total_lik as
		# less than the new-split-hypothesis can't ever win, because
		# the fit is always better with a split.
		# TODO: Proof that this is really correct and leaves minimal
		# amount of hypotheses. And also for variable dt.
		# TODO: The loglikelihood can actually be > 0 with very small
		#	stds!
		new_total_lik = new.total_lik
		for i in range(len(hypotheses)):
			if hypotheses[i].total_lik < new_total_lik:
				break
		else:
			i = None
		hypotheses = hypotheses[:i]
		
		for hypothesis in hypotheses:
			hypothesis.n += 1
			dx = x - hypothesis.mx
			hypothesis.mx += dx/hypothesis.n
			hypothesis.ssx += dx*(x-hypothesis.mx)
			
			dy = y - hypothesis.my
			hypothesis.my += dy/hypothesis.n
			hypothesis.ssy += dy*(y-hypothesis.my)

			# TODO: Verify this! Should be correct, but
			# something weird is going on. Either this is
			# too small or the split likelihood is too large
			# or there's some kind of horrible bug somewhere.
			# Or the logic behind the parameters doesn't work.
			hypothesis.segment_lik = (hypothesis.n*seg_normer - (hypothesis.ssx+hypothesis.ssy)/(2*noise_std**2))
		
		hypotheses.append(new)
		hypotheses.sort(key=lik_comparator)
		
		
	return hypotheses[0].splits
		


	"""
	#import matplotlib.pyplot as plt
	for arg in sweep:
		saccades = func(t, gaze, arg)
		est = reconstruct_fixations(gaze, saccades)
		error = mean_distance(est, pos)
		#plt.cla()
		#ax = plt.subplot(2,1,1)
		#plt.plot(t, est.T[0], color='red')
		#plt.plot(t, pos.T[0], color='green')
		#plt.subplot(2,1,2, sharex=ax)
		#plt.plot(t, norm(est, pos))
		#print error
		#plt.show()
		plt.scatter(arg, -error)
	plt.show()
	"""
		

#def idt

def reconstruct_fixations(gaze, saccades):
	gaze = np.array(gaze)
	idx = np.unique([0] + list(saccades) + [len(gaze)])
	
	new_gaze = np.empty(gaze.shape)
	for i in range(len(idx)-1):
		span = slice(idx[i], idx[i+1])
		span_gaze = gaze[span]
		new_gaze[span] = np.mean(span_gaze, axis=0)
	
	return np.array(new_gaze)


