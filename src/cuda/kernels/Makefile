kernels.ptx: kernels.cu accuracy.impl logistic_loss.impl relu.impl \
	softmax_loss.impl elementwise.impl channel_pooling.impl copy_padded.impl \
	dropout.impl sigmoid.impl l1.impl argmax.impl copy_shifted.impl \
	tanh.impl lrelu.impl stdnormal.impl binary_cross_entropy_loss.impl \
	hinge_loss.impl binary_accuracy.impl
	nvcc -ptx kernels.cu
