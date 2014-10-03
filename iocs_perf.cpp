//#include <armadillo>
#include "segmented_regression.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <ctime>

int main(int argc, char **argv) {
	using Vector = Iocs2d::Vector;
	std::vector<std::string> args(argv+1, argv+argc);
	
	uint n = 1000000;
	if(args.size() > 0) n = std::stoi(args[0]);

	Vector noise_std = {0.0, 0.0};
	if(args.size() > 1) noise_std += std::stod(args[1]);
	else noise_std += 1.0;
	
	double split_rate = 1.0/0.250;
	if(args.size() > 2) split_rate = std::stod(args[2]);
	
	double dt = 1.0/100.0;
	if(args.size() > 3) dt = std::stod(args[3]);

	Iocs2d fitter(noise_std, split_rate);
	
	Vector position = {0.0, 0.0};
	if(args.size() > 4) position += std::stod(args[4]);
	
	fitter.measurement(0, position);
	//std::cerr << "noise_std: " << noise_std << ", split rate: " << split_rate << std::endl;
	auto start_time = clock();
	for(uint i=1; i < n; ++i) {
		/*if(i%1 == 0) {
			std::cout << i << "," << fitter.hypotheses.size() << std::endl;
		}*/
		fitter.measurement(dt, position);
		position += 0.1;
	}

	std::cerr << n/(double(clock()-start_time)/CLOCKS_PER_SEC) << " samples per second" << std::endl;
	std::cerr << fitter.hypotheses.size() << " hypotheses alive" << std::endl;
}
