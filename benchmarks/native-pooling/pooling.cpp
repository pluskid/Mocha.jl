#include <algorithm>
#include <limits>

template <typename T>
void max_pooling_impl(const T *global_input, T *global_output, size_t *global_mask, int width, int height, int channels, int num,
    int pooled_width, int pooled_height, int kernel_w, int kernel_h, int pad_w, int pad_h,
    int stride_w, int stride_h) {

  int input_offset = width*height;
  int output_offset = pooled_width*pooled_height;

  #pragma omp parallel for
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      int offset = (n * channels + c);
      const T *input = global_input + input_offset * offset;
      T *output = global_output + output_offset * offset;
      size_t *mask = global_mask + output_offset * offset;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph*stride_h - pad_h;
          int wstart = pw*stride_w - pad_w;
          int hend   = std::min(hstart + kernel_h, height);
          int wend   = std::min(wstart + kernel_w, width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);

          int pool_index = ph * pooled_width + pw;
          T maxval = -std::numeric_limits<T>::max();
          size_t maxidx = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              int index = h * width + w;
              if (input[index] > maxval) {
                maxval = input[index];
                maxidx = index;
              }
            }
          }
          output[pool_index] = maxval;
          mask[pool_index] = maxidx;
        }
      }
    }
  }
}

extern "C" {
  void max_pooling_impl_double(const double *input, double *output, size_t *mask,
      int width, int height, int channels, int num, int pooled_width, int pooled_height,
      int kernel_w, int kernel_h, int pad_w, int pad_h, int stride_w, int stride_h) {
    max_pooling_impl(input, output, mask, width, height, channels, num,
        pooled_width, pooled_height, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h);
  }
}
