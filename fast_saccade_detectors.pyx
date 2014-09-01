# cython: infer_types
import numpy as np
cimport numpy as np
import cython
import scipy.optimize

def reconstruct_fixations(gaze, saccades):
	gaze = np.array(gaze, dtype=np.float64, copy=True)
	_reconstruct_fixations(gaze, np.array(saccades, dtype=int))
	return gaze

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void _reconstruct_fixations(np.ndarray[np.float64_t, ndim=2] gaze, np.ndarray[np.int_t, ndim=1] saccades):
	cdef np.ndarray[np.int_t] idx = np.unique([0] + list(saccades) + [len(gaze)])
	cdef int n = idx.shape[0]
	cdef int i, s, e, j
	cdef double mx, my
	for i in range(n-1):
		s = idx[i]
		e = idx[i+1]
		for j in range(s, e):
			mx += gaze[j,0]
			my += gaze[j,1]
		mx /= (e-s)
		my /= (e-s)
		for j in range(s, e):
			gaze[j,0] = mx
			gaze[j,1] = my

@cython.boundscheck(False)
@cython.cdivision(True)
def idt(np.ndarray[np.float64_t] t, np.ndarray[np.float64_t, ndim=2]gaze, double threshold=10.0):
	cdef double xmin, xmax, ymin, ymax, x, y
	xmin = ymin = 1.0/0.0
	xmax = ymax = -1.0/0.0
	saccades = []
	for i in range(t.shape[0]):
		x = gaze[i,0]
		y = gaze[i,1]
		xmin = min(x, xmin)
		xmax = max(x, xmax)
		ymin = min(y, ymin)
		ymax = max(y, ymax)
		dispersion = (xmax - xmin) + (ymax - ymin)
		if dispersion > threshold:
			saccades.append(i)
			xmin = ymin = 1.0/0.0
			xmax = ymax = -1.0/0.0
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
	root_hypothesis.mx = gaze[0,0]
	root_hypothesis.my = gaze[0,1]
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
			i += 1
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


def norm(a, b):
	return np.sqrt(np.sum((a - b)**2, axis=1))

def mean_distance(a, b):
	return norm(a, b).mean()

def optimize_1d(func, rng, t, gaze, pos):
	def minimisee(arg):
		saccades = func(t, gaze, arg)
		est = reconstruct_fixations(gaze, saccades)
		return mean_distance(est, pos)
	return scipy.optimize.brute(minimisee, rng)[0]
