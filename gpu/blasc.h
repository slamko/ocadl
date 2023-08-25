#ifndef BLASC_H
#define BLASC_H

#include <stddef.h>

struct mat {
  float *matrix;

  size_t rows;
  size_t cols;
  size_t dim3;
};

#define err(msg) fprintf(stderr, msg)
#define error(msg, ...) fprintf(stderr, msg, __VA_ARGS__)

void mat_print(const struct mat *mat);

size_t mat_mem_size(const struct mat *mat);

struct mat mat3_of_array(float *matrix,
                         size_t rows, size_t cols, size_t dim3);

struct mat mat3_nil(size_t rows, size_t cols, size_t dim3);

struct mat mat3_random(size_t rows, size_t cols, size_t dim3);

struct mat mat_nil(size_t rows, size_t cols);

struct mat mat_make(size_t rows, size_t cols);

struct mat mat_of_array(float *arr, size_t rows, size_t cols);

int mat_add(const struct mat *a, const struct mat *b, struct mat *c);

int mat_sub(const struct mat *a, const struct mat *b, struct mat *c);

int mat_scale(const struct mat *mat, struct mat *res, float scale);

int vec_sum(const struct mat *mat, float *res);

#endif
