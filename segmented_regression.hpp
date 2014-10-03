// TODO! This doesn't agree with the Python-implementation
//	results since porting to Eigen!

#include <vector>
#include <list>
#include <memory>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/StdVector>

typedef unsigned int uint;
using Eigen::Vector2d;
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
	
	
	void measurement(uint i, double dt, double* position) {
		measurement(i, dt, Map<Vector>(position));
	}
	
	void measurement(uint i, double dt, Ref<Vector> position) {
		n++;
		// ARGH! Eigen (as well as Armadillo) does wrong math
		// if we don't eagerly eval!!
		auto delta = (position - mean).eval();
		mean += (delta/n);

		ss += delta.cwiseProduct(position-mean);
		segment_lik = n*model->seg_normer - 0.5*(ss.cwiseProduct(model->noise_prec)).sum();
	}

	double likelihood() const {
		return history_lik + segment_lik;
	}
};

// TODO: Make a general segmented regression thingie when needed
template <uint ndim>
struct Iocs {
	using Vector = Vector2d;
	using Hypothesis = IocsHypothesis<Vector, Iocs>;
	Vector noise_std;
	Vector noise_prec;
	double split_rate;
	// This could maybe be a list
	std::vector<Hypothesis, Eigen::aligned_allocator<Vector>> hypotheses;

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

	void measurement(double dt, Ref<Vector> measurement) {
		if(hypotheses.size() == 0) {
			hypotheses.emplace_back(this);
			auto& root = hypotheses.back();
			root.measurement(0, dt, measurement);
			++i;
			return;
		}
		
		auto& winner = hypotheses.front();
		Hypothesis new_hypo(this, winner, dt, i);
		auto worst_survivor = new_hypo.history_lik;
		
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
		
		// This could maybe be a heap
		std::sort(hypotheses.begin(), hypotheses.end(),
			[](const Hypothesis& a, const Hypothesis& b) {
				return a.likelihood() > b.likelihood();
			}
		);

		//std::cout << i << "," << hypotheses[0].likelihood() << std::endl;
		
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
	
	auto split = fitter.hypotheses.front().splits.tail;
	while(split) {
		saccades[split->value] = 1;
		split = split->parent;
	}
}
