kernels.ptx: kernels.cu accuracy.impl logistic_loss.impl relu.impl \
	softmax_loss.impl elementwise.impl channel_pooling.impl copy_padded.impl
	nvcc -ptx kernels.cu
