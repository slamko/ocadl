#include <CL/opencl.hpp>
#include "ocl.hpp"

extern "C" {
#include "blasc.h"
}

#include <iostream>
#include <fstream>

size_t mat_mem_size(const struct mat *mat) {
    return (mat->rows * mat->cols * mat->dim3 * sizeof(*mat->matrix));
}

void mat_free(struct mat *mat) {
  std::free(mat->matrix);
}

void fatal_error(std::string msg) {
  std::cerr << msg << std::endl;
  std::exit(1);
}

struct mat mat3_of_array(float *matrix, size_t rows, size_t cols, size_t dim3) {
    mat mat = {
        .rows = rows,
        .cols = cols,
        .dim3 = dim3,
    };

    size_t mat_dims = rows * cols * dim3;
    
    mat.matrix = matrix;
    
    return mat;
}

struct mat mat_of_array(float *arr, size_t rows, size_t cols) {
    return mat3_of_array(arr, rows, cols, 1);
}

struct mat mat3_make(size_t rows, size_t cols, size_t dim3) {
    size_t mat_dims = rows * cols;
    
    float *matrix = (float *)std::malloc(mat_dims * sizeof(*matrix));

    if (!matrix) {
        fatal_error("Matrix allocation failed\n");
    }

    return mat3_of_array(matrix, rows, cols, dim3);
}

struct mat mat_make(size_t rows, size_t cols) {
    return mat3_make(rows, cols, 1);
}

struct mat mat3_nil(size_t rows, size_t cols, size_t dim3) {
    size_t mat_dims = rows * cols;
    
    float *matrix = (float *)std::calloc(mat_dims, sizeof *matrix);
    if (!matrix) {
        fatal_error("Matrix allocation failed\n");
    }

    return mat3_of_array(matrix, rows, cols, dim3);
}

struct mat mat_nil(size_t rows, size_t cols) {
    return mat3_nil(rows, cols, 1);
}

void mat_print(mat *mat) {
    for (size_t d3 = 0; d3 < mat->dim3; d3++) {
        for (size_t r = 0; r < mat->rows; r++) {
            for (size_t c = 0; c < mat->cols; c++) {
              std::cout << mat->matrix[c + r * mat->cols + d3 * mat->rows] << "  ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
}

extern "C" int mat_add(const mat *a, const mat *b, mat *c) {
  using namespace cl;
  int ret;

  if (a->rows != b->rows || a->cols != b->cols) {
    return 1;
  }

  *c = mat_make(a->rows, a->cols);
  size_t a_size = mat_mem_size(a);
  size_t b_size = mat_mem_size(b);
  size_t c_size = mat_mem_size(c);

  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;

  Buffer abuf { context, in_flags, a_size, a->matrix };
  Buffer bbuf { context, in_flags, b_size, b->matrix };
  Buffer cbuf { context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, c_size, NULL };

  Kernel kernel { math_prog, "matrix_add" };

  size_t dim1 = c->rows < 16 ? c->rows : align(c->rows, 16);
  size_t dim2 = c->cols < 16 ? c->cols : align(c->cols, 16);

  ret = kernel.setArg(0, abuf);
  if (ret) return ret;
  ret = kernel.setArg(1, bbuf);
  if (ret) return ret;
  ret = kernel.setArg(2, cbuf);
  if (ret) return ret;
  ret = kernel.setArg(3, sizeof(size_t), &dim1);
  if (ret) return ret;
  ret = kernel.setArg(4, sizeof(size_t), &dim2);
  if (ret) return ret;

  size_t ldim1 = c->rows < 16 ? c->rows : 16;
  size_t ldim2 = c->cols < 16 ? c->cols : 16;

  auto glob_range = NDRange(dim1, dim2);
  auto loc_range = NDRange(ldim1, ldim2);

  ret = queue.enqueueNDRangeKernel(kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  ret = queue.enqueueReadBuffer(cbuf, CL_TRUE, 0, c_size, c->matrix);

  return ret;
}

extern "C" int mat_sub(const mat *a, const mat *b, mat *c) {
  using namespace cl;
  int ret = 0;

  if (a->rows != b->rows || a->cols != b->cols) {
    return 1;
  }

  *c = mat_make(a->rows, a->cols);
  size_t a_size = mat_mem_size(a);
  size_t b_size = mat_mem_size(b);
  size_t c_size = mat_mem_size(c);

  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;

  Buffer abuf { context, in_flags, a_size, a->matrix };
  Buffer bbuf { context, in_flags, b_size, b->matrix };
  Buffer cbuf { context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, c_size, NULL };

  Kernel kernel { math_prog, "matrix_sub" };

  size_t dim1 = c->rows < 16 ? c->rows : align(c->rows, 16);
  size_t dim2 = c->cols < 16 ? c->cols : align(c->cols, 16);

  ret |= kernel.setArg(0, abuf);
  ret |= kernel.setArg(1, bbuf);
  ret |= kernel.setArg(2, cbuf);
  ret |= kernel.setArg(3, sizeof(size_t), &dim1);
  ret |= kernel.setArg(4, sizeof(size_t), &dim2);
  if (ret) return ret;

  size_t ldim1 = c->rows < 16 ? c->rows : 16;
  size_t ldim2 = c->cols < 16 ? c->cols : 16;

  auto glob_range = NDRange(dim1, dim2);
  auto loc_range = NDRange(ldim1, ldim2);

  ret = queue.enqueueNDRangeKernel(kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  ret = queue.enqueueReadBuffer(cbuf, CL_TRUE, 0, c_size, c->matrix);

  return ret;
}

extern "C" int mat_scale(const struct mat *mat, struct mat *res, float scale) {
  using namespace cl;
  int ret = 0;

  if (mat->rows == 0 || mat->cols == 0) {
    return 1;
  }

  *res = mat_make(mat->rows, mat->cols);
  size_t mat_size = mat_mem_size(mat);
  size_t res_size = mat_mem_size(res);

  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;

  Buffer mat_buf { context, in_flags, mat_size, mat->matrix };
  Buffer res_buf { context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, res_size, NULL };

  Kernel kernel { math_prog, "matrix_scale" };
  size_t dim1 = mat->rows < 16 ? mat->rows : align(mat->rows, 16);
  size_t dim2 = mat->cols < 16 ? mat->cols : align(mat->cols, 16);

  ret |= kernel.setArg(0, mat_buf);
  ret |= kernel.setArg(1, res_buf);
  ret |= kernel.setArg(2, sizeof(float), &scale);
  ret |= kernel.setArg(3, sizeof(size_t), &dim1);
  ret |= kernel.setArg(4, sizeof(size_t), &dim2);
  if (ret) return ret;

  size_t ldim1 = mat->rows < 16 ? mat->rows : 16;
  size_t ldim2 = mat->cols < 16 ? mat->cols : 16;

  auto glob_range = NDRange(dim1, dim2);
  auto loc_range = NDRange(ldim1, ldim2);

  ret = queue.enqueueNDRangeKernel(kernel, NullRange, glob_range);
  ret = queue.enqueueReadBuffer(res_buf, CL_TRUE, 0, res_size, res->matrix);

  return 0;
}

extern "C" int vec_sum(const struct mat *mat, float *res) {
  *res = 0.0;

  for (size_t i = 0; i < mat->rows * mat->cols; i++) {
    *res += mat->matrix[i];
  }

  return 0;
}
