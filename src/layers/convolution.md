# Implementation Notes of convolution/deconvolution via im2col/col2im

## Tensor shapes

Filter tensor

* convolution: (kernel_w, kernel_h, chann) x (n_filter)
* deconvolution: (kernel_w, kernel_h, n_filter) x (chann)

Col-buffer

* convolution: (w_out, h_out) x (kernel_w,kernel_h, chann)
* deconvolution: (w, h) x (kernel_w,kernel_h, n_filter)

## Convolution

### Convolution Forward

im2col(img) x filter => img_out: (w_out, h_out) x (n_filter)

### Convolution Backward

∂img_out x Tr(filter) => ∂(im2col(img))

Tr(im2col(img)) x ∂img_out => ∂filter

## Deconvolution

### Deconvolution Forward

img x Tr(filter) => im2col(img_out): (w,h) x (kernel_w,kernel_h,n_filter)

### Deconvolution Backward

im2col(∂img_out) x filter => ∂img: (w,h) x chann

Tr(im2col(∂img_out)) x img => ∂filter: (kernel_w,kernel_h,n_filter) x (chann)



# Situation with groups

## Tensor shapes

Filter tensor

* convolution: (kernel_w, kernel_h, chann/n_grp) x (n_filter)
* deconvolution: (kernel_w, kernel_h, n_filter/n_grp) x (chann)

Col-buffer

* convolution: (w_out, h_out) x (kernel_w,kernel_h, chann/n_grp)
* deconvolution: (w,h) x (kernel_w,kernel_h,n_filter/n_grp)

When allocating we do not divide by n_grp because im2col/col2im are done per-image instead of per-group.

## Convolution

### Convolution Forward

im2col(img<grp>) x filter<grp> => img_out<grp>: (w_out,h_out) x (n_filter/n_grp)

### Convolution Backward

∂img_out<grp> x Tr(filter<grp>) => ∂(im2col(img<grp>))

## Deconvolution

### Deconvolution Forward

img<grp> x Tr(filter<grp>) => im2col(img_out<grp>): (w,h) x (kernel_w,kernel_h,n_filter/n_grp)

### Deconvolution Backward

im2col(∂img_out<grp>) x filter<grp> => ∂img<grp>: (w,h) x (chann/n_grp)
