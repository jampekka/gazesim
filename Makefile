DEBUG ?= 0
ifeq ($(DEBUG), 1)
	CFLAGS=-g -O0 -DEIGEN_DONT_VECTORIZE -DEIGEN_DONT_ALIGN_STATICALLY
else
	CFLAGS=-DNDEBUG -Ofast
endif

fast_saccade_detectors.so: fast_saccade_detectors.pyx segmented_regression.hpp
	cython2 $<
	g++ $(CFLAGS) -I. -std=c++11 -shared `python2-config --includes` `pkg-config --cflags --libs eigen3` -o fast_saccade_detectors.so fast_saccade_detectors.c

iocs_perf: iocs_perf.cpp segmented_regression.hpp
	g++ `pkg-config --cflags --libs eigen3` $(CFLAGS) -g -std=c++11 -o $@ -lprofiler -ltcmalloc -Wall $<

.PHONY: perftest
perftest: iocs_perf
	./iocs_perf 1000000 10 1


.PHONY: benchmark
benchmark: benchmark.py fast_saccade_detectors.so gazesim.py saccade_detectors.py
	./benchmark.py

.PHONY: iocs_perf.callgrind
iocs_perf.callgrind: iocs_perf
	valgrind --tool=callgrind --callgrind-out-file=$@ ./$< 100000 0.1 4

.PHONY: profile
profile: iocs_perf.callgrind
	kcachegrind $<
