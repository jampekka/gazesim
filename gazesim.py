from __future__ import division

import itertools

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class JumpingObjectPositionSimulator(object):
	def __init__(self, width=20.0, height=20.0, rate=1.0/0.250):
		self.width = width
		self.height = height
		self.pos = [0.0, 0.0]
		self.rate = rate
		
	
	def __call__(self, dt):
		# In a homogenous Poisson process the probability
		# that there's exactly zero events during dt reduces
		# to exp(-rate * dt). So we can check if we
		# have an event as follows:
		if np.random.rand() < np.exp(-self.rate*dt):
			# No event at this time
			return self.pos, False

		x = (np.random.rand()-0.5)*self.width
		y = (np.random.rand()-0.5)*self.height
		self.pos = [x, y]

		return self.pos, True

class LinearPursuit(object):
	def __init__(self, x0, x1, speed):
		x0 = np.asarray(x0)
		x1 = np.asarray(x1)
		self.endtime = np.linalg.norm(x1-x0)/speed
		self.t = 0.0
		self.x0 = x0
		self.x1 = x1
		

	def __call__(self, dt):
		# NOTE: The caller is responsible for giving
		#	dt of 0 for the "first sample"
		self.t += dt
		if self.t >= self.endtime:
			return self.x1
		
		reltime = self.t/self.endtime
		return self.x0*(1-reltime) + self.x1*reltime

class RandomLinearPursuitSimulator(object):
	def __init__(self, width=20.0, height=20.0, mean_duration=0.250,
			pursuit_speed=scipy.stats.gamma(1.0, scale=10.0).rvs):
		self.width = width
		self.height = height
		self.pos = [0.0, 0.0]
		self.mean_duration = mean_duration
		self.pursuit_speed = pursuit_speed
		self.current_pursuit = LinearPursuit(self.pos, self.pos, pursuit_speed())
	
	@property
	def extent(self):
		return (-self.width*0.5, self.width*0.5), (-self.height*0.5, self.height*0.5)
	
	def __call__(self, dt):
		# In a homogenous Poisson process the probability
		# that there's exactly zero events during dt reduces
		# to exp(-rate * dt). So we can check if we
		# have an event as follows:
		if np.random.rand() < np.exp(-1.0/self.mean_duration*dt):
			# No event at this time
			return self.current_pursuit(dt), False
		
		x = (np.random.rand()-0.5)*self.width
		y = (np.random.rand()-0.5)*self.height
		src = [x, y]
		x = (np.random.rand()-0.5)*self.width
		y = (np.random.rand()-0.5)*self.height
		target = [x, y]
		
		self.current_pursuit = LinearPursuit(src, target, self.pursuit_speed())
		return self.current_pursuit(0.0), True



def constant_accel_saccade(a):
	class saccade(object):
		def __init__(self, init_dt, x0, x1):
			diff = np.subtract(x1, x0)
			length = np.linalg.norm(diff)
			direction = diff/length
			T = np.sqrt(length/a)*2
			
			self.x0 = x0
			self.x1 = x1
			self.direction = direction
			self.T = T
			
			self.duration = T
			self.mean_speed = length/T
			self.max_speed = a*(T/2.0)
			self.max_accel = a
		
	
		def __iter__(self):
			x0 = self.x0
			T = self.T
			direction = self.direction

			t = yield x0
			while t < T/2.0:
				dist = 0.5*a*t**2
				t += yield x0 + dist*direction
	
			while t < T:
				dist = 1/4*a*(-2*t**2 + 4*t*T - T**2)
				t += yield x0 + dist*direction
	
	
			yield self.x1
	
	return saccade
		


def gaussian_noiser(sx=1.0, sy=1.0):
	def noiser(dt, xy):
		x = xy[0]+np.random.randn()*sx
		y = xy[1]+np.random.randn()*sy
		return [x,y]
	return noiser

def infrange(i=0):
	while True:
		yield i
		i += 1

def generate_sequence(simulator, noiser, sampling_rate=500.0):
	dt = 1.0/sampling_rate
	t = 0.0
	while True:
		pos, had_saccade = simulator(dt)
		gaze = noiser(dt, pos)
		yield t, pos, gaze, had_saccade
		
		t += dt

def test():
	sampling_rate = 100.0
	duration = 60.0
	simulator = BallisticObjectSimulator(rate=1.0/1.0)
	noiser = gaussian_noiser()
	generator = generate_sequence(simulator, noiser)
	#print list(itertools.islice(generator, int(duration*sampling_rate)))
	t, pos, gaze, saccades = zip(*itertools.islice(generator, int(duration*sampling_rate)))
	plt.plot(t, zip(*pos)[0])
	plt.plot(t, zip(*gaze)[0], '.')
	plt.show()

	
def plot_main_sequence():
	origin = np.zeros(2)
	direction = np.ones(2)/np.sqrt(2)
	saccade_gen=constant_accel_saccade(9000.0)

	d = []
	for dist in np.arange(0.01, 100.0):
		saccade = saccade_gen(0, origin, dist*direction)
		d.append([dist, saccade.max_speed])
	
	plt.loglog(*zip(*d))
	plt.show()

def dummy_main_series():
	#main_series = lambda distance: distance*5
	main_series = lambda distance: 10*distance
	distances = np.linspace(0.01, 100, 1000)
	
	plt.subplot(2,1,1)
	plt.plot(distances, main_series(distances))
	plt.subplot(2,1,2)
	plt.loglog(distances, main_series(distances))
	plt.show()
	

if __name__ == '__main__':
	#test()
	#plot_main_sequence()
	#dummy_main_series()

	import time
	sim = RandomLinearPursuitSimulator()
	noiser = gaussian_noiser()
	noiser = lambda dt, pos: pos
	
	#x, y = zip(*xy)
	#plt.plot(x)
	#plt.xlim(-0.5*sim.width, 0.5*sim.width)
	#plt.ylim(-0.5*sim.width, 0.5*sim.width)
	plt.xlim(sim.extent[0])
	plt.ylim(sim.extent[1])
	plt.ion()
	
	npoints = 10
	pointstack = []
	prev_gaze = None
	prev_t = time.time()
	gazes = []
	thruths = []
	times = []
	rate = 60.0
	
	cross, = plt.plot(0, 0, 'o')
	for i in range(10000):
		#t = time.time()
		t = i/rate
		times.append(t)
		dt = t - prev_t
		prev_t = t
		pos, is_split = sim(dt)
		gaze = noiser(dt, pos)
		gazes.append(gaze)
		thruths.append(pos)
		
		cross.set_data(pos[0], pos[1]); plt.pause(1.0/rate)
		continue
		for gp in pointstack:
			gp.set_alpha(gp.get_alpha()*0.8)
		if prev_gaze is not None:
			gp, = plt.plot([gaze[0], prev_gaze[0]], [gaze[1], prev_gaze[1]], '-', color='red', alpha=1.0, markersize=6, linewidth=4)
			pointstack.append(gp)
		prev_gaze = gaze
		if len(pointstack) > npoints:
			gp = pointstack.pop(0)
			gp.remove()
		
		plt.pause(1.0/rate)
	
	plt.clf()
	plt.ioff()
	plt.plot(times, zip(*thruths)[0], color='green')
	plt.plot(times, zip(*gazes)[0], color='red')

	plt.show()
