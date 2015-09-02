#define THREADS_PER_BLOCK_X 128
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_Z 8

#define LOG_THRESHOLD 1e-20

#include "elementwise.impl"
#include "copy_padded.impl"
#include "copy_shifted.impl"

#include "logistic_loss.impl"
#include "softmax_loss.impl"
#include "accuracy.impl"
#include "channel_pooling.impl"
#include "dropout.impl"
#include "argmax.impl"
#include "index2onehot.impl"

#include "relu.impl"
#include "lrelu.impl"
#include "sigmoid.impl"
#include "tanh.impl"

#include "l1.impl"
