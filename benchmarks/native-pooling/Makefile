bm: libpooling.so
	julia pooling-bm.jl

libpooling.so: pooling.cpp
	g++ -shared -o libpooling.so -fPIC -Wall -O3 pooling.cpp
