from __future__ import division

import numpy as np
import scipy.optimize, scipy.stats
import operator

def norm(a, b):
	return np.sqrt(np.sum((a - b)**2, axis=1))

def mean_distance(a, b):
	return norm(a, b).mean()

def erode_consecutive(idx):
	return idx[:-1][np.diff(idx) > 1]

def reconstruct_pursuits(t, gaze, saccades):
	# TODO: Make this fast
	idx = np.unique([0] + list(saccades) + [len(gaze)])
	print saccades, idx
	
	import matplotlib.pyplot as plt
	result = np.empty(gaze.shape)
	n = idx.shape[0]
	for i in range(n-1):
		slc = slice(idx[i], idx[i+1])
		my_t = t[slc]
		#print len(my_t)
		fit = np.polyfit(my_t, gaze[slc], 1)
		result[slc] = fit[0]*my_t + fit[1]
	
	return result


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

def iocs(ts, gaze, split_rate=1.0/0.250, noise_std=1.0):
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
	split_lik = lambda dt: 2*np.log(1 - np.exp(-split_rate*dt))
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
		for si in range(len(hypotheses)):
			if hypotheses[si].total_lik < new_total_lik:
				break
		else:
			si = None
		hypotheses = hypotheses[:si]
		
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
			#print len(hypotheses)
		
		hypotheses.append(new)
		hypotheses.sort(key=lik_comparator)
		#print "%s,%s"%(i, hypotheses[0].total_lik)
		
		
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
# TODO! Iols is about 4x as split-happy as IOCS. Probably due to not
# 	discounting the extra degree of any way
def iols(ts, gaze, split_rate=1.0/0.250/4, noise_std=np.array([1.0, 1.0])):
	ndim = len(gaze[0])
	# TODO: Handle the axes separately
	seg_normer = np.log(1.0/(noise_std.prod()*np.sqrt(np.pi*2)**ndim))
	
	def segment_logpdf(h):
		# TODO! Just sanity checking!
		#return (h.n*seg_normer - (h.pos_ss)/(2*noise_std**2)).sum()
		
		# Calculate the residual.
		# One way to rationalize this is to think
		# that Pearson correlation:
		# r = co_ss/(sqrt(pos_ss)*sqrt(t_ss))
		# And coefficient of determination is
		# r**2 = 1 - residual_ss/pos_ss = co_ss**2/(pos_ss*t_ss)
		# from which the residual_ss is:
		residual_ss = h.pos_ss - h.co_ss**2/h.t_ss
		return hypo.n*seg_normer - (0.5*residual_ss/noise_std**2).sum()

	root = SplitHypothesis()
	root.splits = []
	root.history_lik = 0.0
	root.segment_lik = seg_normer
	root.n = 1
	root.pos_m = gaze[0].copy()
	root.t_m = ts[0].copy()

	root.pos_ss = 0.0
	root.t_ss = 0.0
	root.co_ss = 0.0
	
	split_lik = lambda dt: ndim*np.log(1 - np.exp(-split_rate*dt))

	hypotheses = [root]
	prev_t = ts[0]
	for i, (t, pos) in enumerate(zip(ts, gaze)[1:], 1):
		dt = t - prev_t
		prev_t = t
		
		winner = max(hypotheses, key=lambda h: h.total_lik)
		new = SplitHypothesis()
		new.history_lik = winner.total_lik + split_lik(dt)
		new.segment_lik = seg_normer
		new.n = 1
		new.splits = winner.splits + [i]
		new.pos_m = pos.copy()
		new.t_m = ts[0].copy()
		new.pos_ss = 0.0
		new.t_ss = 0.0
		new.co_ss = 0.0
		
		new_total_lik = new.total_lik
		hypotheses = [h for h in hypotheses if h.total_lik >= new_total_lik]

		for hypo in hypotheses:
			hypo.n += 1

			d_pos = pos - hypo.pos_m
			# Calculate hypothesis position mean incrementally
			hypo.pos_m += d_pos/hypo.n
			# And incremental sum of squares
			hypo.pos_ss += d_pos*(pos - hypo.pos_m)
			
			# Above with s/pos/t/g
			d_t = t - hypo.t_m
			hypo.t_m += d_t/hypo.n
			hypo.t_ss += d_t*(t - hypo.t_m)

			# Calculate the regression SS incrementally
			# (for both independent axes simultaneously)
			hypo.co_ss += ((hypo.n-1)/hypo.n)*d_pos*d_t
			
			
			hypo.segment_lik = segment_logpdf(hypo)
		
		
		hypotheses.append(new)
		#print len(hypotheses)
	winner = max(hypotheses, key=lambda h: h.total_lik)
	return winner.splits

def iols_noprune(ts, gaze, split_rate=1.0/0.250, noise_std=np.array([1.0, 1.0])):
	ndim = len(gaze[0])
	# TODO: Handle the axes separately
	seg_normer = np.log(1.0/(noise_std.prod()*np.sqrt(np.pi*2)**ndim))
	
	def segment_logpdf(h):
		
		# Calculate the residual.
		# One way to rationalize this is to think
		# that Pearson correlation:
		# r = co_ss/(sqrt(pos_ss)*sqrt(t_ss))
		# And coefficient of determination is
		# r**2 = 1 - residual_ss/pos_ss = co_ss**2/(pos_ss*t_ss)
		# from which the residual_ss is:
		residual_ss = h.pos_ss - h.co_ss**2/h.t_ss
		return hypo.n*seg_normer - (0.5*residual_ss/noise_std**2).sum()

	root = SplitHypothesis()
	root.splits = []
	root.history_lik = 0.0
	root.segment_lik = seg_normer
	root.n = 1
	root.pos_m = gaze[0].copy()
	root.t_m = ts[0].copy()

	root.pos_ss = 0.0
	root.t_ss = 0.0
	root.co_ss = 0.0
	
	split_lik = lambda dt: ndim*np.log(1 - np.exp(-split_rate*dt))

	hypotheses = [root]
	prev_t = ts[0]
	for i, (t, pos) in enumerate(zip(ts, gaze)[1:], 1):
		dt = t - prev_t
		prev_t = t
		
		winner = max(hypotheses, key=lambda h: h.total_lik)
		# TODO! Something wrong here!
		new = SplitHypothesis()
		new.history_lik = winner.total_lik + split_lik(dt)
		new.segment_lik = seg_normer
		new.n = 1
		new.splits = winner.splits + [i]
		new.pos_m = pos.copy()
		new.t_m = ts[0].copy()
		new.pos_ss = 0.0
		new.t_ss = 0.0
		new.co_ss = 0.0
		
		new_total_lik = new.total_lik
		#hypotheses = [h for h in hypotheses if h.total_lik >= new_total_lik]

		for hypo in hypotheses:
			hypo.n += 1

			d_pos = pos - hypo.pos_m
			# Calculate hypothesis position mean incrementally
			hypo.pos_m += d_pos/hypo.n
			# And incremental sum of squares
			hypo.pos_ss += d_pos*(pos - hypo.pos_m)
			
			# Above with s/pos/t/g
			d_t = t - hypo.t_m
			hypo.t_m += d_t/hypo.n
			hypo.t_ss += d_t*(t - hypo.t_m)

			# Calculate the regression SS incrementally
			# (for both independent axes simultaneously)
			hypo.co_ss += ((hypo.n-1)/hypo.n)*d_pos*d_t
			
			
			hypo.segment_lik = segment_logpdf(hypo)
		
		
		hypotheses.append(new)
		#print len(hypotheses)
	winner = max(hypotheses, key=lambda h: h.total_lik)
	return winner.splits


def reconstruct_fixations(gaze, saccades):
	gaze = np.array(gaze)
	idx = np.unique([0] + list(saccades) + [len(gaze)])
	
	new_gaze = np.empty(gaze.shape)
	for i in range(len(idx)-1):
		span = slice(idx[i], idx[i+1])
		span_gaze = gaze[span]
		new_gaze[span] = np.mean(span_gaze, axis=0)
	
	return np.array(new_gaze)


