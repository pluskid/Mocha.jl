bm: libextim2col.so
	LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH julia im2col-bm.jl

libextim2col.so: im2col.c
	gcc -shared -o libextim2col.so -fPIC -Wall -std=c99 -O3 im2col.c
