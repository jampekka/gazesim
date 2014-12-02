// TODO: Ugly copypaste stuff. Fix when brainpower returns.

#include <vector>
#include <list>
#include <forward_list>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <cmath>

typedef unsigned int uint;
using Eigen::Array;
using Eigen::Map;
using Eigen::Ref;

template <typename T>
struct SharedList {
	struct SharedNode {
		SharedNode *parent = NULL;
		uint refcount;
		T value;

		SharedNode(SharedNode *parent, T value)
			:parent(parent), refcount(1), value(value)
		{
			if(parent) {
				parent->refcount += 1;
			}
		}

		~SharedNode() {
			if(parent) {
				parent->refcount -= 1;
			}
		}
	};

	SharedNode *tail;
	
	SharedList() {
		tail = NULL;
	}

	SharedList(const SharedList &parent, T value)
	{
		tail = new SharedNode(parent.tail, value);
	}
	
	SharedList(const SharedList& that) {
		tail = that.tail;
		if(tail) {
			tail->refcount += 1;
		}
	}

	~SharedList() {
		if(!tail) return;
		#warning "Make sure this doesn't leak nor crash!"
		tail->refcount -= 1;
		
		// We could in theory do this cascade in the
		// destructor of the SharedNode. However, in practice
		// even on realistic data that leads to a huge recursion that
		// leads to a stack overflow. And this is probably a lot faster
		// anyway.
		auto node = tail;
		while(node and node->refcount == 0) {
			auto killme = node;
			node = node->parent;
			delete killme;
		}
	}
};

template <class Vector, class Model>
struct IocsHypothesis {
	int n = 0;
	Vector mean;
	Vector ss;
	
	// This should probably be const Model* const, but STL tears a new one
	// if we declare it so, as it will kill the move constructor.
	// Actually it should be a reference, but STL really really doesn't
	// like that. This is one horrible language.
	const Model *model = NULL;
	SharedList<uint> splits;
	double history_lik = 0.0;
	//double segment_lik = 0.0;
	double _total_likelihood = 0.0;

	
	IocsHypothesis(Model* model, IocsHypothesis& parent, double dt, uint i)
		:model(model), splits(parent.splits, i)
	{
		history_lik = parent.likelihood();
		history_lik += model->split_likelihood(dt);
		mean *= 0.0;
		ss *= 0.0;
	}
	
	IocsHypothesis(Model* model)
		:model(model)
	{}
	
	
	inline void measurement(uint i, double dt, double* position) {
		measurement(i, dt, Map<Vector>(position));
	}
	
	void measurement(uint i, double dt, Ref<Vector> position) {
		n++;
		// ARGH! Eigen (as well as Armadillo) does wrong math
		// if we don't eagerly eval!!
		auto delta = (position - mean).eval();
		mean += (delta/n);

		ss += delta.cwiseProduct(position-mean);
		auto segment_lik = n*model->seg_normer - 0.5*(ss.cwiseProduct(model->noise_prec)).sum();
		_total_likelihood = history_lik + segment_lik;
	}

	double likelihood() const {
		return _total_likelihood;
		//return history_lik + segment_lik;
	}
};

// TODO: Make a general segmented regression thingie when needed
template <uint ndim>
struct Iocs {
	using Vector = Array<double, ndim, 1>;
	using Hypothesis = IocsHypothesis<Vector, Iocs>;
	Vector noise_std;
	Vector noise_prec;
	double split_rate;
	// std::vector<Hypothesis, Eigen::aligned_allocator<Vector>> hypotheses;
	// At least the boost pool allocators are quite slow, although
	// almost half of the performance seems to go to malloc
	std::forward_list<Hypothesis, Eigen::aligned_allocator<Vector>> hypotheses;

	double seg_normer;

	uint i = 0;

        Iocs(Vector noise_std, double split_rate)
		:noise_std(noise_std), split_rate(split_rate)
	{
		seg_normer = log(1.0/(pow(sqrt(2*M_PI), ndim)*noise_std.prod()));
		// There should be cwiseSquare, but there isn't.
		noise_prec = 1.0/(noise_std.cwiseProduct(noise_std).array());
        }
	
	double split_likelihood(double dt) const {
		return ndim*log(1.0-exp(-split_rate*dt));
	}
	
	void measurement(double dt, double *position) {
		Map<Vector> pos(position);
		measurement(dt, pos);
	}

	Hypothesis& get_winner() {
		static const auto likcmp = [](const Hypothesis& a, const Hypothesis& b) {
			return a.likelihood() <= b.likelihood();
		};

		return *std::max_element(hypotheses.begin(), hypotheses.end(), likcmp);
	}

	void measurement(double dt, Ref<Vector> measurement) {
		if(hypotheses.empty()) {
			hypotheses.emplace_front(this);
			auto& root = hypotheses.front();
			root.measurement(0, dt, measurement);
			++i;
			return;
		}
		

		auto& winner = get_winner();
		hypotheses.emplace_front(this, winner, dt, i);
		auto& new_hypo = hypotheses.front();
		auto worst_survivor = new_hypo.history_lik;
		
		static const auto no_chance = [&worst_survivor](const Hypothesis& hypo) {
			return hypo.likelihood() < worst_survivor;
		};
		
		hypotheses.remove_if(no_chance);
		//hypotheses.push_front(new_hypo);

		for(auto& hypo: hypotheses) {
			hypo.measurement(i, dt, measurement);
		}

		++i;
	}
};

typedef Iocs<2u> Iocs2d;

void iocs2d(double *ts, double *gaze, uint length,
		double *noise_std, double split_rate, int *saccades) {
	static const auto ndim = 2u;
	if(length == 0) return;

	Iocs2d::Vector noise_std_vec(noise_std);
	Iocs2d fitter(noise_std_vec, split_rate);

	auto prev_t = ts[0];
	for(uint i=0; i < length; ++i) {
		fitter.measurement(ts[i] - prev_t, &gaze[i*ndim]);
		prev_t = ts[i];
	}
	
	auto split = fitter.get_winner().splits.tail;
	while(split) {
		saccades[split->value] = 1;
		split = split->parent;
	}
}

/* Naive optimal linear segmentation.
   Assumes that velocity between segments can be
   "infinite" (ie start of the segment doesn't in any way depend from the
   previous). Not very realistic, but statistically a simple case.

   TODO: Probably not even optimal!
   TODO: The Vector should actually be an Array
*/ 
template <class Vector, class Model>
struct NolsHypothesis {
	int n = 0;
	Vector mean_x;
	Vector ss_x;
	double t;
	double mean_t;
	double ss_t;
	Vector ss_xt;
	
	// This should probably be const Model* const, but STL tears a new one
	// if we declare it so, as it will kill the move constructor.
	// Actually it should be a reference, but STL really really doesn't
	// like that. This is one horrible language.
	const Model *model = NULL;
	SharedList<uint> splits;
	double history_lik = 0.0;
	//double segment_lik = 0.0;
	double _total_likelihood = 0.0;
	
	void __initialize_variables() {
		// C++ constructors suck.
		t *= 0;
		mean_t *= 0;
		mean_x *= 0;
		ss_x *= 0;
		ss_t *= 0;
		ss_xt *= 0;
	}
	
	NolsHypothesis(Model* model, NolsHypothesis& parent, double dt, uint i)
		:model(model), splits(parent.splits, i)
	{
		history_lik = parent.likelihood();
		history_lik += model->split_likelihood(dt);
		__initialize_variables();
	}
	
	NolsHypothesis(Model* model)
		:model(model)
	{
		__initialize_variables();
	}
	
	
	inline void measurement(uint i, double dt, double* position) {
		measurement(i, dt, Map<Vector>(position));
	}
	
	void measurement(uint i, double dt, Ref<Vector> position) {
		n++;
		// ARGH! Eigen (as well as Armadillo) does wrong math
		// if we don't eagerly eval!!
		auto delta_x = (position - mean_x).eval();
		mean_x += (delta_x/n);
		ss_x += delta_x.cwiseProduct(position-mean_x);

		t += dt;
		auto delta_t = t - mean_t;
		mean_t += delta_t/n;
		ss_t += delta_t*(t - mean_t);

		
		// Calculate the regression SS incrementally
		// (for both independent axes simultaneously)
		ss_xt += ((n-1)/double(n))*delta_x*delta_t;
		
		auto residual_ss = (ss_x - ss_xt.pow(2)/ss_t).eval();
		if(ss_t == 0) residual_ss = ss_x; // Sometimes zero by zero is zero
		auto likelihoods = n*(1.0/(model->noise_std*std::sqrt(2*M_PI))).log() -
				residual_ss/(2*model->noise_std.pow(2));
		
		/*std::cout << "Residual and x ss's" << std::endl;
		std::cout << ss_xt << std::endl;
		std::cout << ss_x << std::endl;*/

		auto segment_lik = likelihoods.sum();
		//std::cout << n << " " << ss_t << " " << " " << segment_lik << std::endl;
		_total_likelihood = history_lik + segment_lik;
		
	}

	double likelihood() const {
		return _total_likelihood;
		//return history_lik + segment_lik;
	}
};

// TODO: Make a general segmented regression thingie when needed
template <uint ndim>
struct Nols {
	using Vector = Array<double, ndim, 1>;
	using Hypothesis = NolsHypothesis<Vector, Nols>;
	Vector noise_std;
	Vector noise_prec;
	double split_rate;
	// std::vector<Hypothesis, Eigen::aligned_allocator<Vector>> hypotheses;
	// At least the boost pool allocators are quite slow, although
	// almost half of the performance seems to go to malloc
	std::forward_list<Hypothesis, Eigen::aligned_allocator<Vector>> hypotheses;

	double seg_normer;

	uint i = 0;

        Nols(Vector noise_std, double split_rate)
		:noise_std(noise_std), split_rate(split_rate)
	{
		seg_normer = log(1.0/(pow(sqrt(2*M_PI), ndim)*noise_std.prod()));
		// There should be cwiseSquare, but there isn't.
		noise_prec = 1.0/(noise_std.cwiseProduct(noise_std).array());
        }
	
	double split_likelihood(double dt) const {
		return ndim*log(1.0-exp(-split_rate*dt));
	}
	
	void measurement(double dt, double *position) {
		Map<Vector> pos(position);
		measurement(dt, pos);
	}

	Hypothesis& get_winner() {
		static const auto likcmp = [](const Hypothesis& a, const Hypothesis& b) {
			return a.likelihood() <= b.likelihood();
		};

		return *std::max_element(hypotheses.begin(), hypotheses.end(), likcmp);
	}

	void measurement(double dt, Ref<Vector> measurement) {
		if(hypotheses.empty()) {
			hypotheses.emplace_front(this);
			auto& root = hypotheses.front();
			root.measurement(0, dt, measurement);
			++i;
			return;
		}
		

		auto& winner = get_winner();
		hypotheses.emplace_front(this, winner, dt, i);
		auto& new_hypo = hypotheses.front();
		auto worst_survivor = new_hypo.history_lik;
		
		static const auto no_chance = [&worst_survivor](const Hypothesis& hypo) {
			return hypo.likelihood() < worst_survivor;
		};
		
		hypotheses.remove_if(no_chance);
		//hypotheses.push_front(new_hypo);

		for(auto& hypo: hypotheses) {
			hypo.measurement(i, dt, measurement);
		}

		++i;
	}
};

/*struct Nols {
	
}*/

typedef Nols<2u> Nols2d;

void nols2d(double *ts, double *gaze, uint length,
		double *noise_std, double split_rate, int *saccades) {
	static const auto ndim = 2u;
	if(length == 0) return;

	Nols2d::Vector noise_std_vec(noise_std);
	Nols2d fitter(noise_std_vec, split_rate);

	auto prev_t = ts[0];
	for(uint i=0; i < length; ++i) {
		fitter.measurement(ts[i] - prev_t, &gaze[i*ndim]);
		prev_t = ts[i];
	}
	
	auto split = fitter.get_winner().splits.tail;
	while(split) {
		saccades[split->value] = 1;
		split = split->parent;
	}
}
