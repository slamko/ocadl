#include <CL/opencl.hpp>
#include <iostream>
#include "ocl.hpp"

extern "C" {
#include "blasc.h"
  #include "deep.h"
}

extern "C" int fully_connected_bp(
                       const struct mat *weight_mat,
                       const struct mat *prev_act_vec,
                       const struct mat *act_vec,
                       const struct mat *diff_vec,
                       struct mat *prev_diff_vec,
                       struct mat *wgrad_mat,
                       struct mat *bgrad_vec,
                       long actf,
                       _Bool prev_layer) {
  using namespace cl;
  int ret = 0;

  if (act_vec->cols != weight_mat->cols
      || weight_mat->rows != prev_act_vec->cols) {
    return 1;
  }

  *prev_diff_vec = mat_nil(1, weight_mat->rows);
  // *wgrad_mat     = mat_make(weight_mat->rows, weight_mat->cols);
  // *bgrad_vec     = mat_make(1, act_vec->cols);
  
  size_t prev_act_size = mat_mem_size(prev_act_vec);
  size_t act_size = mat_mem_size(act_vec);
  size_t diff_size = mat_mem_size(diff_vec);
  size_t wmat_size = mat_mem_size(weight_mat);
  
  size_t prev_diff_size = mat_mem_size(prev_diff_vec);
  size_t wgrad_size = mat_mem_size(wgrad_mat);
  size_t bgrad_size = mat_mem_size(bgrad_vec);

  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;

  Buffer wmat_buf { context, in_flags, wmat_size, weight_mat->matrix };
  Buffer prev_act_buf { context, in_flags, prev_act_size, prev_act_vec->matrix };
  Buffer act_buf { context, in_flags, act_size, act_vec->matrix };
  Buffer diff_buf { context, in_flags, diff_size, diff_vec->matrix };

  Buffer prev_diff_buf { context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, prev_diff_size, NULL };
  Buffer wgrad_buf { context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY, wgrad_size, wgrad_mat->matrix };
  Buffer bgrad_buf { context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY, bgrad_size, bgrad_vec->matrix };

  Kernel kernel { nn_prog, "dense_bp" };
  size_t width = weight_mat->cols;
  size_t global_work_size [1] = { width };

  size_t ldim1 = 32;
  size_t dim1 = align(width, ldim1);

  cl_ulong n = weight_mat->rows;

  struct mat cache_mat = mat_make(n, act_vec->cols);
  size_t cache_size = mat_mem_size(&cache_mat);
  Buffer cache_buf { context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, cache_size, NULL };

  ret |= kernel.setArg(0, wmat_buf);
  ret |= kernel.setArg(1, prev_act_buf);
  ret |= kernel.setArg(2, act_buf);
  ret |= kernel.setArg(3, diff_buf);
  ret |= kernel.setArg(4, sizeof(cl_ulong), &n);
  ret |= kernel.setArg(5, sizeof(cl_ulong), &width);

  ret |= kernel.setArg(6, cache_buf);
  ret |= kernel.setArg(7, wgrad_buf);
  ret |= kernel.setArg(8, bgrad_buf);
  ret |= kernel.setArg(9, sizeof(cl_long), &actf);

  if (ret) return ret;

  auto glob_range = NDRange(dim1);
  auto loc_range = NDRange(ldim1);

  ret = queue.enqueueNDRangeKernel(kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  ret |= queue.enqueueReadBuffer(cache_buf, CL_TRUE, 0, cache_size, cache_mat.matrix);
  // ret |= queue.enqueueReadBuffer(prev_diff_buf, CL_TRUE, 0, prev_diff_size, prev_diff_vec->matrix);
  ret |= queue.enqueueReadBuffer(wgrad_buf, CL_TRUE, 0, wgrad_size, wgrad_mat->matrix);
  ret |= queue.enqueueReadBuffer(bgrad_buf, CL_TRUE, 0, bgrad_size, bgrad_vec->matrix);
  if (ret) return ret;

  if (prev_layer) {
    for (size_t i = 0; i < n; i++) {
      prev_diff_vec->matrix[i] = 0.0;
      for (size_t j = 0; j < cache_mat.cols; j++) {
        prev_diff_vec->matrix[i] +=
          cache_mat.matrix[i * cache_mat.cols + j];
      }
    }
  }

  return ret;
}


extern "C" int fully_connected_ff(const struct mat *input,
                                  const struct mat *weight_mat,
                                  const struct mat *bias_vec,
                                  struct mat *res,
                                  long actf) {
  using namespace cl;

  cl_int ret = {0};
  
  if (input->cols != weight_mat->rows
      || weight_mat->cols != bias_vec->cols) {
    return 1;
  }
  
  *res = mat_nil(1, weight_mat->cols);
  
  size_t inp_mat_size = mat_mem_size(input);
  size_t wmat_size = mat_mem_size(weight_mat);
  size_t bmat_size = mat_mem_size(bias_vec);
  size_t res_mat_size = mat_mem_size(res);
  
  Kernel kernel { nn_prog, "dense_ff" };
  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
  
  Buffer inp_buf { context, in_flags, inp_mat_size, input->matrix };
  Buffer wmat_buf { context, in_flags, wmat_size, weight_mat->matrix };
  Buffer bmat_buf { context, in_flags, bmat_size, bias_vec->matrix };
  
  Buffer res_buf { context, CL_MEM_WRITE_ONLY, res_mat_size, NULL };
  
  cl_ulong mat_dim = weight_mat->rows;
  cl_ulong width = weight_mat->cols;

  ret |= kernel.setArg(0, inp_buf);
  ret |= kernel.setArg(1, wmat_buf);
  ret |= kernel.setArg(2, bmat_buf);
  ret |= kernel.setArg(3, res_buf);
  ret |= kernel.setArg(4, sizeof(cl_ulong), &mat_dim);
  ret |= kernel.setArg(5, sizeof(cl_ulong), &width);
  ret |= kernel.setArg(6, sizeof(cl_long), &actf);

  if (ret) return ret;
  
  cl_ulong dim = weight_mat->cols;
  size_t ldim1 = 32;
  size_t dim1 = align(dim, ldim1);
 
  auto glob_range = NDRange(dim1);
  auto loc_range = NDRange(ldim1);

  ret = queue.enqueueNDRangeKernel(kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  ret = queue.enqueueReadBuffer(res_buf, CL_TRUE, 0, res_mat_size, res->matrix);

  return ret;
}


extern "C" int conv2d_ff(const struct mat *input,
                         const struct mat *weight_mat,
                         const struct mat *bias_vec,
                         struct mat *res) {
  using namespace cl;

  cl_int ret = {0};
  
  if (input->cols != weight_mat->rows
      || weight_mat->cols != bias_vec->cols) {
    return 1;
  }
  
  *res = mat_nil(1, weight_mat->cols);
  
  size_t inp_mat_size = mat_mem_size(input);
  size_t wmat_size = mat_mem_size(weight_mat);
  size_t bmat_size = mat_mem_size(bias_vec);
  size_t res_mat_size = mat_mem_size(res);
  
  Kernel kernel { nn_prog, "dense_ff" };
  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
  
  Buffer inp_buf { context, in_flags, inp_mat_size, input->matrix };
  Buffer wmat_buf { context, in_flags, wmat_size, weight_mat->matrix };
  Buffer bmat_buf { context, in_flags, bmat_size, bias_vec->matrix };
  SVMAllocator<float, SVMTraitCoarse<>> svm_alloc { context };
  // svm_alloc.
  Buffer res_buf { context, CL_MEM_WRITE_ONLY, res_mat_size, NULL };
  
  cl_ulong mat_dim = weight_mat->rows;
  cl_ulong width = weight_mat->cols;

  ret |= kernel.setArg(0, inp_buf);
  ret |= kernel.setArg(1, wmat_buf);
  ret |= kernel.setArg(2, bmat_buf);
  ret |= kernel.setArg(3, res_buf);
  ret |= kernel.setArg(4, sizeof(cl_ulong), &mat_dim);
  ret |= kernel.setArg(5, sizeof(cl_ulong), &width);

  if (ret) return ret;
  
  cl_ulong dim = weight_mat->cols;
  size_t ldim1 = 32;
  size_t dim1 = align(dim, ldim1);
 
  auto glob_range = NDRange(dim1);
  auto loc_range = NDRange(ldim1);

  ret = queue.enqueueNDRangeKernel(kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  ret = queue.enqueueReadBuffer(res_buf, CL_TRUE, 0, res_mat_size, res->matrix);

  return ret;
}


