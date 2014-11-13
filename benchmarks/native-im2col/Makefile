bm: libextim2col.so
	julia im2col-bm.jl

libextim2col.so: im2col.c
	gcc -shared -o libextim2col.so -fPIC -Wall -std=c99 -O3 im2col.c
