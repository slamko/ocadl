#include <iostream>
#include "ocl.hpp"
#include <string>
extern "C" {
#include "blasc.h"
#include "ocl.h"
}
#include <immintrin.h>
#include <xmmintrin.h>

void sse_conv(const struct mat *image,
              const struct mat *kernel,
              unsigned long res_width,
              unsigned long res_height,
              struct mat *res) {

  *res = mat_nil(align(res_height, 4), align(res_width, 4));

  for (size_t h = 0; h < res_height; h++) {
    for (size_t w = 0; w < res_width; w++) {
      __m128 sum = _mm_set1_ps(0.0);

      for (size_t r = 0; r < kernel->rows; r++) {
          __m128 data = _mm_loadu_ps(image->matrix + h * image->cols + w + r * image->cols);
          __m128 kern_row = _mm_load_ps(kernel->matrix + r * kernel->cols);
          __m128 cur_res = _mm_mul_ps(data, kern_row);
          sum = _mm_add_ps(sum, cur_res);
      }

      // printf("Back\n");
      __m128 hsum = _mm_hadd_ps(sum, sum);
      __m128 res_val = _mm_hadd_ps(hsum, hsum);
      _mm_store_ss(res->matrix + h * res_width + w, res_val);
    }
  }
}

int main(int argc, char **argv) {
  using namespace std::chrono;

  std::srand(std::time(nullptr));

  int ret = ocl_init();

  if (argc != 3) {
    std::cerr << "Args: dim kdim\n";
    std::exit(1);
  }

  char *kdim_str = argv[2];
  char *dim_str = argv[1];

  int dim = atoi(dim_str);
  int kdim = atoi(kdim_str);

  struct mat m1 = mat3_random(dim, dim, 1);
  struct mat kernel = mat3_random(kdim, kdim, 1);
  struct mat res1 = {0};
  struct mat res2 = {0};

  // mat_print(&m1);
  auto start = high_resolution_clock::now();
  // conv2(&m1, &kernel, 0, dim - kdim + 1, dim - kdim + 1, &res1);
  // convolve(&m1, &kernel, 0, dim - kdim + 1, dim - kdim + 1, &res2);
  sse_conv(&m1, &kernel, dim - kdim + 1, dim - kdim + 1, &res2);
  auto stop = high_resolution_clock::now();
  auto dur = duration_cast<microseconds>(stop - start);
  std::cout << "Time: " << dur.count() << std::endl;

  /*
  printf("Innov\n");
  mat_print(&res1);
  printf("Trad\n");
  mat_print(&res2);
  */

  ocl_finish();
}
