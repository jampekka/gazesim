import numpy as np, matplotlib.pyplot as plt
import scipy.integrate

A = 40000.0
B = A

# HACK!
class accelerator(object):
	def __init__(self):
		self.autopilot = False
		self.end_t = 0.0
		self.prev_update = -1.0
		self.update_freq = 0.001
	
	def __call__(self, t, (x, v), pf):
		if t - self.prev_update > self.update_freq:
			#x += np.random.randn()*0.1
			self.prev_update = t
			d = pf(t) - x
			#if np.abs(d) < 0.01:
			#	return 0.0
			self.A = A*np.sign(d)
			foo = np.sqrt(v**2 + self.A*d)
			mid_dur = (-v - foo)/self.A
			if mid_dur < 0:
				mid_dur = (-v + foo)/self.A
			mid_dur *= 1.1
			self.mid_t = t + mid_dur
			self.end_t = t + mid_dur*2

		tdiff = t - self.mid_t
		#if t > self.end_t:
		#	return 0.0
		return self.A*np.tanh(-tdiff*50)

		
		return 
		p = pf(t)
		d = p - x
		
		print A - 2*v**2/(d+1)
		return A - 2*v**2/(d+1)
		#predicted_x = x + d + A*np.tanh(-predicted_overshoot)*B*(d/v)**2

		if v > 1e-6:
			t_left = d/v
			predicted_x = x + v*t_left - 0.5*B*t_left**2
			predicted_overshoot = predicted_x - p
		else:
			predicted_overshoot = -d
	
		a = A*np.tanh(-predicted_overshoot)
		#print a
		return a

accel = accelerator()

#accel = lambda t, x, bt=0.5: A*np.sign(bt-t)

def nth_order_ode(func, *args):
	i = [0]
	def a(y, x):
		diff = func(t, y, *args)
		# The API is insane!
		diffs = np.concatenate((y[1:], [diff]))
		return diffs
	return a

def eval_ode(func, ts, init, *args):
	state = np.zeros(len(init)+1)
	state[:-1] = init
	res = [state.copy()]
	ode = nth_order_ode(func, *args)
	t = ts[0]
	for dt in np.diff(ts):
		t += dt
		state[-1] = func(t, state[:-1], *args)
		state[:-1] += state[1:]*dt
		#print state
		res.append(np.copy(state))
	return np.array(res)
		
	"""
	solver = scipy.integrate.ode(nth_order_ode(func, *args))
	solver.set_integrator('dop853')
	solver.set_initial_value(init, 0.0)
	
	def god_damn_this_api_is_horrible():
		for t in ts:
			yield solver.integrate(t)
	return np.array(list(god_damn_this_api_is_horrible()))
	"""
	return scipy.integrate.odeint(nth_order_ode(func, args), init, t)


def test():
	t = np.arange(0, 0.1, 0.0001)
	
	x, v, a = eval_ode(accel, t, [0.0, 0.0], lambda t: 10.0).T
	#print x
	plt.subplot(3,1,1)
	plt.plot(t*1000, x)
	plt.subplot(3,1,2)
	plt.plot(t*1000, v)
	plt.subplot(3,1,3)
	plt.plot(t*1000, a)
	plt.show()

def deriv_profile(deg, locfunc, t=np.arange(0, 0.5, 0.001), mags=np.arange(1, 30, 5)):
	for mag in mags:
		pfunc = lambda t: np.ones(np.size(t))*mag*(t > 0.01)
		x = locfunc(t, pfunc)
		dt = np.gradient(t)
		for i in range(deg):
			x = np.gradient(x)/dt
		
		plt.plot(t, x)

def quadratic_damper(A=100000.0, D=1000.0):
	def accel(t, (x, v), pos):
		d = pos(t)-x
		return -A*d**2

	return accel

def impulse_response(locfunc, t=np.arange(0, 0.5, 0.001)):
	def impulsehack(t):
		x = np.zeros(np.size(t))
		x[10] = 1.0
		return x

	plt.plot(t, locfunc(t, impulsehack))

import scipy.signal
from scipy.signal import lfilter, butter, cheby1

def stupid_lti(freq=10.0):
	def filt(t, pos):
		nyq = 1/(2*np.diff(t).mean())
		print nyq
		#b, a = cheby1(1, 1.0, freq/nyq)
		b, a = scipy.signal.bessel(2, freq/nyq)
		print b, a
		return lfilter(b, a, pos(t))	

	return filt



t = np.arange(0.0, 10000.0, 0.01)

#plt.plot(t, (1-np.exp(-t)))
#plt.plot(t, np.exp(-t))
#plt.show()

#gauss = lambda x: np.exp(-x**2)
#plt.axvline(0)
plt.show()

#speed_profile(lambda t, posf: eval_ode(accel, t, [0.0, 0.0], posf))
#speed_profile(lambda t, posf: eval_ode(quadratic_damper(), t, [0.0, 0.0], posf).T[0])
plt.subplot(3,1,1)
deriv_profile(0, stupid_lti())
plt.subplot(3,1,2)
deriv_profile(2, stupid_lti())
plt.subplot(3,1,3)
impulse_response(stupid_lti())
plt.show()
