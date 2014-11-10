#define THREADS_PER_BLOCK_X 128
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_Z 8

#define LOG_THRESHOLD 1e-20

#include "logistic_loss.impl"
#include "softmax_loss.impl"

#include "accuracy.impl"

#include "relu.impl"

