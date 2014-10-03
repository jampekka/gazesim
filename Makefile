fast_saccade_detectors.so: fast_saccade_detectors.pyx segmented_regression.hpp
	cython2 $<
	g++ -Ofast  -I. -std=c++11 -shared `python2-config --includes` -larmadillo -DARMA_NO_DEBUG -o fast_saccade_detectors.so fast_saccade_detectors.c

iocs_perf: iocs_perf.cpp segmented_regression.hpp
	g++ -larmadillo -Ofast -g -std=c++11 -DARMA_NO_DEBUG -o $@ -lprofiler -ltcmalloc -Wall $<

.PHONY: perftest
perftest: iocs_perf
	./iocs_perf 1000000 10 1


.PHONY: benchmark
benchmark: benchmark.py fast_saccade_detectors.so gazesim.py saccade_detectors.py
	./benchmark.py

iocs_perf.callgrind: iocs_perf
	valgrind --tool=callgrind --callgrind-out-file=$@ ./$< 100000 0.1 4

.PHONY: profile
profile: iocs_perf.callgrind
	kcachegrind $<
