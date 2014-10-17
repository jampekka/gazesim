import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import itertools

def idm(vmax=500.0, amax=1000.0, bmax=None, e=4.0, smin=0.0, T=0.0):
	if bmax is None:
		bmax = amax
	def accel(xs, xt, vs, vt=0.0):
		dv = vs - vt
		dx = xs - xt
		s = smin + vs*T + vs*dv/(2*np.sqrt(amax*bmax))
		a = amax*(1 - (vs/vmax)**e - (s/dx)**2)
		
		return a
	
	def accel(xs, xt, vs, vt=0.0):
		dv = vs - vt
		dx = xs - xt
		a = -amax*dx
		
		return a
	
	
	return accel

def nth_odediffs(f, n):
	def diffs(p, t):
		y[-1] = f(p, t)
		return y
	return diffs

def simulator(dt, p, x, w, r=0.7):
	t = 0.0
	v = 0.0
	vmax = 300.0
	
	accel = idm()

	amax = 60000.0
	damping = 200.0
	xmax = 100
	
	def sigmoid(x):
		return 1.0/(1.0 + np.exp(-x))

	def af(x):
		#return accel(x, p, v)
		#return 100*(1.0/(x+p)**2 - 1.0/(x-p)**2)
		d = x-p
		#return amax*(sigmoid(-d) - sigmoid(d)) - damping*v
		#return -10*(np.exp(d/10.0) - np.exp(-d/10.0)) - damping*v
		#return amax*(1 - (2)/(1+np.exp(-0.5*d))) - damping*v
		return -amax*np.tanh(d) - damping*v

		#return amax*(-np.exp(-np.exp(-d/5.0)) + np.exp(-np.exp(d/5.0))) - damping*v
	rng = np.linspace(-20, 20, 100); plt.plot(rng, af(rng+p))
	plt.show()
	
	while True:
		ndt = yield t, x
		if ndt is not None:
			dt = ndt
		a = af(x)
		v += a*dt
		x += v*dt
		t += dt

def simulator(dt, p, x, w, r=0.7):
	t = 0.0
	v = 0.0
	vmax = 300.0
	
	accel = idm()

	amax = A = 60000.0
	B = 1.0
	damping = D = 200.0
	xmax = 100
	v0 = 0.0
	
	def sigmoid(x):
		return 1.0/(1.0 + np.exp(-x))
	
	tanh, exp = np.tanh, np.exp
	def vf(t, x):
		#return accel(x, p, v)
		#return 100*(1.0/(x+p)**2 - 1.0/(x-p)**2)
		d = x-p
		#return amax*(sigmoid(-d) - sigmoid(d)) - damping*v
		#return -10*(np.exp(d/10.0) - np.exp(-d/10.0)) - damping*v
		#return amax*(1 - (2)/(1+np.exp(-0.5*d))) - damping*v
		#return -amax*np.tanh(d) - damping*v

		#return amax*(-np.exp(-np.exp(-d/5.0)) + np.exp(-np.exp(d/5.0))) - damping*v
		#return -((A*tanh(d*B) - A*tanh(d*B)*exp(-t*D))/D + v0*exp(-t*D))
		return -(exp(-t*D)*(A*tanh((x-p)*B)*exp(t*D)+v0*D-A*tanh((x-p)*B)))/D
	#rng = np.linspace(-20, 20, 100); plt.plot(rng, af(rng+p))
	#plt.show()
	
	while True:
		ndt = yield t, x
		if ndt is not None:
			dt = ndt
		v = vf(t, x)
		x += v*dt
		t += dt

def brute(func, simmer, rng):
	for v in rng:
		x = np.array(list(simmer(v)))
		plt.plot(*x.T)
		yield v, func(x)


def saccade_x(t, A=1000.0, B=1.0, D=0.001, p=10.0):
	exp = np.exp
	log = np.log
	sqrt = np.sqrt
	sinh = np.sinh
	arcsinh = np.arcsinh
	return p-(arcsinh(exp((A*B*(1-exp(-(D*t))-D*t))/D**2)*sinh(B*p)))/B

# Parameters very crudely estimated from plots in the
# Bahill et al 1980 article "Variability and development of a normative
# data base for saccadic eye movements"
def gaze_accel(max_accel=80000.0, max_speed=684.0, B=0.1):
	A = max_accel
	D = A/max_speed
	tanh = np.tanh
	def a(t, (x, v), p):
		return -A*tanh(B*(x-p(t))) - D*v
	#a.A = A
	#a.B = B
	#a.D = D
	#a.saccade_peak_accel = lambda d: -A*tanh(B*d)
	return a

def nth_order_ode(func):
	i = [0]
	def a(y, t, *args):
		diff = func(t, y, *args)
		# The API is insane!
		diffs = np.concatenate((y[1:], [diff]))
		return diffs
	return a

accel_model = gaze_accel()

def saccader(t, p, x0=0.0, v0=0.0):
	return scipy.integrate.odeint(nth_order_ode(accel_model), [x0, v0], t, (p,))[:,0]

def simulate_saccade(d, dt=0.00001, maxdur=0.1, gazefunc=saccader):
	t = np.arange(0, maxdur, dt)
	p = lambda t: d
	return t, gazefunc(t, p)

mags = np.linspace(1, 50, 5)

saccades = np.array(map(simulate_saccade, mags))

t = saccades[0][0]
print t.shape
dt = np.mean(np.diff(t))
positions = saccades[:,1]
speeds = np.array([np.gradient(s) for s in positions])/dt
accels = np.array([np.gradient(s) for s in speeds])/dt

plt.subplot(3,1,1)
for x in positions: plt.plot(t*1000, x)
plt.subplot(3,1,2)
for x in speeds: plt.plot(t*1000, x)
plt.subplot(3,1,3)
for x in accels: plt.plot(t*1000, x)
plt.show()

plt.subplot(2,1,2)
plt.plot(mags, np.max(accels, axis=1), 'r', label="Numerical estimate")
#plt.plot(mags, -accel_model.saccade_peak_accel(mags) , 'r--', label="Analytical")
plt.ylabel('Peak accel (deg/s**2)')

plt.subplot(2,1,1)
plt.plot(mags, np.max(speeds, axis=1), 'b', label="Numerical estimate")
plt.ylabel('Peak speed (deg/s)')
# Estimate of Bahill et al 1980
plt.plot(mags, 684*(1-np.exp(-0.1*mags)), 'b--', label="Bahil et al")
plt.legend(loc=4)
plt.show()

#xrng = np.linspace(-10, 10, 100)
#plt.plot(xrng, 1/(xrng+10) - xrng+10 + 1/(xrng-10) - xrng-10)
#plt.show()

#sim = simulator(0.01, 0.0, 10.0, 10.0)
dt = 0.001
simmer = lambda d: itertools.islice(simulator(dt, d, 0, 100.0), 1000)

xt = lambda x, n: np.diff(x[:,1], n)/np.diff(x[:,0])[n-1]

"""
drng = np.linspace(0.1, 90, 100)
maxspeeds = list(brute(
	lambda x: np.max(xt(x, 1)),
	simmer, drng))

maxspeeds = np.array(maxspeeds)
#print maxspeeds
plt.show()
plt.plot(*(maxspeeds).T)

#sig = itertools.islice(sim, 1000)

plt.show()
"""
