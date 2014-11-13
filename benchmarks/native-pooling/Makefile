bm: libpooling.so
	julia pooling-bm.jl

# On OS X, official clang does not support OpenMP yet, see http://clang-omp.github.io/
# If failed to compile, please remove "-fopenmp" below.
libpooling.so: pooling.cpp
	g++ -shared -o libpooling.so -fPIC -Wall -O3 -fopenmp pooling.cpp
