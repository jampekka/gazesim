#include <vector>
#include <memory>
#include <cmath>
#include <armadillo>
#include <algorithm>
#include <iostream>

namespace ar = arma;
typedef unsigned int uint;

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
		tail->refcount -= 1;
		
		// We could in theory do this cascade in the
		// destructor of the SharedNode. However, in practice
		// even on realistic data that leads to a huge recursion that
		// leads to a stack overflow. And this is probably a lot faster
		// anyway.
		auto node = tail;
		while(node and node->refcount == 0) {
			if(node->refcount == 0) {
				auto killme = node;
				node = node->parent;
				delete killme;
			}
		}
	}
};

template <class Vector, class Model>
struct IocsHypothesis {
	// TODO: We can save a lot of memory and allocations by doing
	//	a shared pointer linked list thingie.
	int n = 0;
	Vector mean = {0.0, 0.0};
	Vector ss = {0.0, 0.0};
	
	// This should probably be const Model* const, but STL tears a new one
	// if we declare it so, as it will kill the move constructor.
	// Actually it should be a reference, but STL really really doesn't
	// like that. This is one horrible language.
	const Model *model = NULL;
	SharedList<uint> splits;
	double history_lik = 0.0;
	double segment_lik = 0.0;

	
	IocsHypothesis(Model* model, IocsHypothesis& parent, double dt, uint i)
		:model(model), splits(parent.splits, i)
	{
		history_lik = parent.likelihood();
		history_lik += model->split_likelihood(dt);
	}

	IocsHypothesis(Model* model)
		:model(model)
	{}

	void measurement(uint i, double dt, Vector position) {
		n++;
		// We need an explicit eval here, otherwise armadillo
		// calculates a wrong value.
		auto delta = (position - mean).eval();
		mean += (delta/n);

		// Percent sign is a element wise multiplication.
		// I'd like NumPy-style array and matrix separation better.
		// TODO: Hmm.. using * here doesn't trigger an error compile time,
		// so there must be some performance penalties involved.
		ss += delta % (position-mean);
		segment_lik = n*model->seg_normer -
			0.5*
			ar::sum(
				ss%model->noise_prec
				);

	}

	double likelihood() const {
		return history_lik + segment_lik;
	}
};

// TODO: Make a general segmented regression thingie when needed
template <uint ndim>
struct Iocs {
	using Vector = ar::vec::fixed<ndim>;
	using Hypothesis = IocsHypothesis<Vector, Iocs>;
	Vector noise_std;
	Vector noise_prec;
	double split_rate;
	std::vector<Hypothesis> hypotheses;

	double seg_normer;

	uint i = 0;

        Iocs(Vector noise_std, double split_rate)
		:noise_std(noise_std), split_rate(split_rate)
	{
		seg_normer = log(1.0/(pow(sqrt(2*M_PI), ndim)*ar::prod(noise_std)));
		noise_prec = 1.0/ar::square(noise_std);
        }

	double split_likelihood(double dt) const {
		return ndim*log(1.0-exp(-split_rate*dt));
	}

	void measurement(double dt, const Vector& measurement) {
		if(hypotheses.size() == 0) {
			hypotheses.emplace_back(this);
			auto& root = hypotheses.back();
			root.measurement(0, dt, measurement);
			++i;
			return;
		}
		
		auto& winner = hypotheses[0];
		Hypothesis new_hypo(this, winner, dt, i);
		auto worst_survivor = new_hypo.history_lik + 1e-6;
		
		auto cutoff = std::find_if(hypotheses.begin(), hypotheses.end(),
			[&worst_survivor](const Hypothesis& hypo) {
				return hypo.likelihood() <= worst_survivor;
				}
			);
		
		if(cutoff != hypotheses.end()) {
			hypotheses.erase(cutoff, hypotheses.end());
		}

		hypotheses.push_back(new_hypo);

		for(auto& hypo: hypotheses) {
			hypo.measurement(i, dt, measurement);
		}
		
		std::sort(hypotheses.begin(), hypotheses.end(),
			[](const Hypothesis& a, const Hypothesis& b) {
				return a.likelihood() > b.likelihood();
			}
		);

		//std::cout << i << "," << hypotheses[0].likelihood() << std::endl;
		
		++i;
	}
};

/*
template <class Vector>
std::vector<uint> iocs(
		std::vector<double>& ts,
		std::vector< Vector >& gaze,
	        Vector noise_std = 1.0,
		double split_rate=1.0/0.250)
{
        using GazeVector = StaticVector<double, ndim>;
	using Hypothesis = IocsHypothesis<GazeVector>;
	std::vector< Hypothesis > hypotheses;

	auto seg_normer = log(1.0/(pow(sqrt(2*blaze::M_PI), ndim)*noise_std.prod()));
	auto split_lik = [&split_rate](double dt) {
		return ndim*log(1.0-exp(-split_rate*dt));
	};
	auto likelihood_cmp =  [](Hypothesis& a, Hypothesis& b) {
		return a.likelihood() > b.likelihood();
	};

	Hypothesis root_hypo;
	root_hypo.n = 1;
	root_hypo.mean = gaze;
	//root_hypo.ss = {0.0}; Let's hope it's zero by default
	root_hypo.history_lik = 0.0;
	root_hypo.segment_lik = seg_normer;
	
	hypotheses.push_back(root_hypo);
	
	auto prev_t = ts[0];

	for(uint i = 1; i < ts.size(); ++i) {
		auto t = ts[i];
		auto g = gaze[i];
		auto dt = t - prev_t; prev_t = t;
		
		auto winner = hypotheses[0];

		Hypothesis new_hypo;
		std::copy(winner.splits.begin(), winner.splits.end(),
			new_hypo.splits.begin());
		new_hypo.n = 1;
		new_hypo.mean = gaze;
		//new_hypo.ss = {0.0}; Let's hope it's zero by default
		new_hypo.history_lik = winner.likelihood() + split_lik(dt);
		new_hypo.segment_lik = seg_normer;
		

		auto best_survivor = new_hypo.likelihood();
		auto itr = hypotheses.begin();
		for(; itr < hypotheses.end(); itr++) {
			if(itr->likelihood() < best_survivor) break;
		}
		if(itr != hypotheses.end()) {
			hypotheses.resize(itr-hypotheses.begin());
		}

		for(auto hypothesis: hypotheses) {
			hypothesis.n += 1;
			auto delta = g - hypothesis.mean;
			hypothesis.mean += delta/hypothesis.n;
			hypothesis.ss += delta*(g-hypothesis.mean);
			
			hypothesis.segment_lik = hypothesis.n*seg_normer -
				(hypothesis.ss*(1.0/noise_std)).sum()*0.5;
		}

		hypotheses.push_back(new_hypo);
		std::sort(hypotheses.begin(), hypotheses.end(), likelihood_cmp);

	}

	return hypotheses[0].splits;

}*/

/*
void iocs2d(double *ts, double *gaze, uint length,
			double noise_std, double split_rate) {
		
		StaticVector<double, 2> noise_std_vec;
                noise_std_vec *= 0.0;
		iocs<2>(
			Map<Array<double, Dynamic, 1> >(ts, length, 1),
			Map<Array<double, Dynamic, 2> >(gaze, length, 2),
			noise_std_vec, split_rate);
}*/

typedef Iocs<2u> Iocs2d;

void iocs2d(double *ts, double *gaze, uint length,
		double *noise_std, double split_rate, int *saccades) {
	static const auto ndim = 2u;
	if(length == 0) return;

	Iocs2d::Vector noise_std_vec(noise_std);
	Iocs2d fitter(noise_std_vec, split_rate);

	auto prev_t = ts[0];
	for(uint i=0; i < length; ++i) {
		fitter.measurement(ts[i] - prev_t, Iocs2d::Vector(&gaze[i*ndim]));
		prev_t = ts[i];
	}
	
	auto split = fitter.hypotheses[0].splits.tail;
	while(split) {
		saccades[split->value] = 1;
		split = split->parent;
	}
}
