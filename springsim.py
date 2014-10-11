import numpy as np
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

	

def simulator(dt, p, x, w, r=0.7):
	t = 0.0
	v = 0.0
	vmax = 300.0
	
	accel = idm()

	amax = 60000.0
	damping = 100.0
	xmax = 90
	def af(x):
		#return accel(x, p, v)
		#return 100*(1.0/(x+p)**2 - 1.0/(x-p)**2)
		d = x-p
		return amax*(1 - (2)/(1+np.exp(-0.5*d))) - damping*v
	
	#rng = np.linspace(-20, 20, 100); plt.plot(rng, af(rng))
	#plt.show()
	
	while True:
		ndt = yield t, x
		if ndt is not None:
			dt = ndt
		a = af(x)
		v += a*dt
		x += v*dt
		t += dt


def brute(func, simmer, rng):
	for v in rng:
		x = np.array(list(simmer(v)))
		plt.plot(*x.T)
		yield v, func(x)


#sim = simulator(0.01, 0.0, 10.0, 10.0)
dt = 0.001
simmer = lambda d: itertools.islice(simulator(dt, d, 0, 100.0), 1000)

xt = lambda x, n: np.diff(x[:,1], n)/np.diff(x[:,0])[n-1]

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
