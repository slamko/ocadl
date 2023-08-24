#include <stddef.h>

struct mat {
  float *matrix;

  size_t rows;
  size_t cols;
  size_t dim3;
};

struct mat mat3_of_array(float *matrix,
                         size_t rows, size_t cols, size_t dim3);

struct mat mat3_nil(size_t rows, size_t cols, size_t dim3);

struct mat mat3_random(size_t rows, size_t cols, size_t dim3);

struct mat mat_nil(size_t rows, size_t cols);

struct mat mat_make(size_t rows, size_t cols);

struct mat mat_of_array(float *arr, size_t rows, size_t cols);

int mat_add(const struct mat *a, const struct mat *b, struct mat *c);
