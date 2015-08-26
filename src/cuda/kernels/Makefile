kernels.ptx: kernels.cu accuracy.impl logistic_loss.impl relu.impl \
	softmax_loss.impl elementwise.impl channel_pooling.impl copy_padded.impl \
	dropout.impl sigmoid.impl l1.impl argmax.impl copy_shifted.impl \
	tanh.impl lrelu.impl
	nvcc -ptx kernels.cu
