#include <CL/opencl.hpp>
#include <iostream>

#include "ocl.hpp"
#include "blasc.hpp"

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
                       int prev_layer) {
  using namespace cl;
  int ret = 0;

  if (act_vec->cols != weight_mat->cols
      || weight_mat->rows != prev_act_vec->cols) {
    return 1;
  }

  *prev_diff_vec = mat_nil(1, weight_mat->rows);
  
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
  cl_ulong n = weight_mat->rows;
  cl_ulong width = weight_mat->cols;

  size_t global_work_size [1] = { width };

  size_t ldim1 = 8;
  size_t ldim2 = 16;
  size_t dim1 = align(width, ldim1);
  size_t dim2 = n;

  Matrix cache_mat { mat_make(n, act_vec->cols) };
  size_t cache_size = mat_mem_size(&cache_mat.matrix);
  Buffer cache_buf { context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, cache_size, NULL };

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

  auto glob_range = NDRange(dim1, dim2);
  auto loc_range = NDRange(ldim1, ldim2);

  ret = queue.enqueueNDRangeKernel(kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  // ret |= queue.enqueueReadBuffer(cache_buf, CL_TRUE, 0, cache_size, cache_mat.matrix.matrix);
  // ret |= queue.enqueueReadBuffer(prev_diff_buf, CL_TRUE, 0, prev_diff_size, prev_diff_vec->matrix);
  ret |= queue.enqueueReadBuffer(wgrad_buf, CL_TRUE, 0, wgrad_size, wgrad_mat->matrix);
  ret |= queue.enqueueReadBuffer(bgrad_buf, CL_TRUE, 0, bgrad_size, bgrad_vec->matrix);
  if (ret) return ret;

  if (prev_layer) {
    Kernel sum_kern { nn_prog, "mat_sum" };
    int argi = 0;
    kern_set_arg(sum_kern, cache_buf);
    kern_set_size_arg(sum_kern, sizeof(cl_ulong), &n);
    kern_set_size_arg(sum_kern, sizeof(cl_ulong), &width);
    kern_set_arg(sum_kern, prev_diff_buf);
    
    auto sum_glob_range = NDRange(align(n, 128));
    auto sum_loc_range = NDRange(128);
    ret = queue.enqueueNDRangeKernel(sum_kern, NullRange, sum_glob_range, sum_loc_range);
    if (ret) return ret;
    ret |= queue.enqueueReadBuffer(prev_diff_buf, CL_TRUE, 0, prev_diff_size, prev_diff_vec->matrix);
  }

  return ret;
}

extern "C" int conv_bp(const struct mat *prev_act_mat,
                       const struct mat *kernels_mat,
                       const struct mat *act_mat,
                       const struct mat *diff_mat,
                       struct mat *prev_diff_mat,
                       struct mat *kern_grad_mat,
                       struct mat *bgrad_vec,
                       long actf,
                       unsigned long stride,
                       unsigned long padding,
                       int prev_layer) {
  using namespace cl;
  int ret = 0;

  if (diff_mat->cols != act_mat->cols ||
      diff_mat->rows != act_mat->rows ||
      act_mat->dim3 != kernels_mat->dim3 ||
      act_mat->dim3 != diff_mat->dim3 ||
      kernels_mat->rows > prev_act_mat->rows ||
      kernels_mat->cols > prev_act_mat->cols ||
      kern_grad_mat->rows != kernels_mat->rows ||
      kern_grad_mat->cols != kernels_mat->cols ||
      kern_grad_mat->dim3 != kernels_mat->dim3) {
    return 1;
  }

  *prev_diff_mat= mat3_nil(prev_act_mat->rows, prev_act_mat->cols, prev_act_mat->dim3);
  
  size_t prev_act_size = mat_mem_size(prev_act_mat);
  size_t act_size = mat_mem_size(act_mat);
  size_t diff_size = mat_mem_size(diff_mat);
  size_t wmat_size = mat_mem_size(kernels_mat);
  
  size_t prev_diff_size = mat_mem_size(prev_diff_mat);
  size_t kern_grad_size = mat_mem_size(kern_grad_mat);
  size_t bgrad_size = mat_mem_size(bgrad_vec);

  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;

  Buffer kerns_buf { context, in_flags, wmat_size, kernels_mat->matrix };
  Buffer prev_act_buf { context, in_flags, prev_act_size, prev_act_mat->matrix };
  Buffer act_buf { context, in_flags, act_size, act_mat->matrix };
  Buffer diff_buf { context, in_flags, diff_size, diff_mat->matrix };

  Buffer prev_diff_buf { context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, prev_diff_size, NULL };
  Buffer kern_grad_buf { context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY, kern_grad_size, kern_grad_mat->matrix };
  Buffer bgrad_buf { context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_READ_ONLY, bgrad_size, bgrad_vec->matrix };

  Kernel deriv_kernel { nn_prog, "conv_deriv_bp" };
  Matrix dz_mat { mat_make(act_mat->rows, act_mat->cols) };
  size_t dz_size = mat_mem_size(&dz_mat.matrix);
  Buffer dz_buf { context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, dz_size, NULL };

  int argi = 0;
  kern_set_arg(deriv_kernel, act_buf);
  kern_set_arg(deriv_kernel, diff_buf);
  kern_set_size_arg(deriv_kernel, sizeof(cl_ulong), &actf);
  kern_set_size_arg(deriv_kernel, sizeof(cl_ulong), &act_mat->cols);
  kern_set_size_arg(deriv_kernel, sizeof(cl_ulong), &act_mat->rows);
  kern_set_arg(deriv_kernel, dz_buf);
  
  size_t ldim1 = 8;
  size_t ldim2 = 16;
  size_t dim1 = align(act_mat->cols, ldim1);
  size_t dim2 = align(act_mat->rows, ldim2);

  if (ret) return ret;

  auto glob_range = NDRange(dim1, dim2, act_mat->dim3);
  auto loc_range = NDRange(ldim1, ldim2, 1);

  ret = queue.enqueueNDRangeKernel(deriv_kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  Kernel bp_kernel { nn_prog, "conv_bp" };
  argi = 0;
  kern_set_arg(bp_kernel, prev_act_buf);
  kern_set_arg(bp_kernel, dz_buf);

  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &stride);
  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &padding);
  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &actf);

  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &act_mat->cols);
  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &act_mat->rows);
  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &act_mat->dim3);

  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &prev_act_mat->cols);
  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &prev_act_mat->rows);
  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &prev_act_mat->dim3);

  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &kernels_mat->cols);
  kern_set_size_arg(bp_kernel, sizeof(cl_ulong), &kernels_mat->rows);
  
  ldim1 = 8;
  ldim2 = 16;
  dim1 = align(kernels_mat->cols, ldim1);
  dim2 = align(kernels_mat->rows, ldim2);

  if (ret) return ret;

  glob_range = NDRange(dim1, dim2, kernels_mat->dim3);
  loc_range = NDRange(ldim1, ldim2, 1);

  ret = queue.enqueueNDRangeKernel(bp_kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  ret |= queue.enqueueReadBuffer(kern_grad_buf, CL_TRUE, 0, kern_grad_size, kern_grad_mat->matrix);
 
  /*
  if (prev_layer) {
    Kernel sum_kern { nn_prog, "mat_sum" };
    int argi = 0;
    kern_set_arg(sum_kern, cache_buf);
    kern_set_size_arg(sum_kern, sizeof(cl_ulong), &n);
    kern_set_size_arg(sum_kern, sizeof(cl_ulong), &width);
    kern_set_arg(sum_kern, prev_diff_buf);
    
    auto sum_glob_range = NDRange(align(n, 128));
    auto sum_loc_range = NDRange(128);
    ret = queue.enqueueNDRangeKernel(sum_kern, NullRange, sum_glob_range, sum_loc_range);
    if (ret) return ret;
    ret |= queue.enqueueReadBuffer(prev_diff_buf, CL_TRUE, 0, prev_diff_size, prev_diff_vec->matrix);
  }
  */

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

extern "C" int conv_ff(const struct mat *input,
                       const struct mat *kernels,
                       const struct mat *bias_vec,
                       long actf,
                       unsigned long padding,
                       unsigned long stride,
                       unsigned long res_width,
                       unsigned long res_height,
                       struct mat *res) {
  
  using namespace cl;

  cl_int ret = {0};
  
  if (kernels->dim3 != bias_vec->dim3) {
    return 1;
  }
  
  *res = mat3_make(res_width, res_height, kernels->dim3);
  
  size_t inp_mat_size = mat_mem_size(input);
  size_t kern_vec_size = mat_mem_size(kernels);
  size_t bmat_size = mat_mem_size(bias_vec);
  size_t res_mat_size = mat_mem_size(res);
  
  Kernel kernel { nn_prog, "conv_ff" };
  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
  
  Buffer inp_buf { context, in_flags, inp_mat_size, input->matrix };
  Buffer kern_vec_buf { context, in_flags, kern_vec_size, kernels->matrix };
  Buffer bmat_buf { context, in_flags, bmat_size, bias_vec->matrix };
  Buffer res_buf { context, CL_MEM_WRITE_ONLY, res_mat_size, NULL };
  
  cl_ulong xdim = input->cols;
  cl_ulong ydim = input->rows;
  cl_ulong zdim = res->dim3;

  size_t argi = 0;
  kern_set_arg(kernel, inp_buf);
  kern_set_arg(kernel, kern_vec_buf);
  kern_set_arg(kernel, bmat_buf);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &stride);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &padding);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &actf);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &input->cols);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &input->rows);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &kernels->dim3);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &input->dim3);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &kernels->cols);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &kernels->rows);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &res_width);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &res_height);

  kern_set_arg(kernel, res_buf);

  if (ret) return ret;
  
  size_t ldim1 = 32, ldim2 = 32, ldim3 = 1;

  size_t dim1 = align(xdim, ldim1);
  size_t dim2 = align(ydim, ldim2);
  size_t dim3 = align(zdim, ldim3);
 
  auto glob_range = NDRange(dim1, dim2, dim3);
  auto loc_range = NDRange(ldim1, ldim2, ldim3);

  ret = queue.enqueueNDRangeKernel(kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  ret = queue.enqueueReadBuffer(res_buf, CL_TRUE, 0, res_mat_size, res->matrix);

  return ret;
}

extern "C" int pooling_ff(const struct mat *input,
                          long type,
                          unsigned long stride,
                          unsigned long res_width,
                          unsigned long res_height,
                          unsigned long filter_width,
                          unsigned long filter_height,
                          struct mat *res) {
  using namespace cl;

  cl_int ret = {0};
  
  if (!input->matrix) {
    return 1;
  }
  
  *res = mat3_make(res_width, res_height, input->dim3);
  
  size_t inp_mat_size = mat_mem_size(input);
  size_t res_mat_size = mat_mem_size(res);
  
  Kernel kernel { nn_prog, "pooling_ff" };
  cl_mem_flags in_flags =  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS;
  
  Buffer inp_buf { context, in_flags, inp_mat_size, input->matrix };
  Buffer res_buf { context, CL_MEM_WRITE_ONLY, res_mat_size, NULL };
  
  cl_ulong xdim = res->cols;
  cl_ulong ydim = input->rows;
  cl_ulong zdim = res->dim3;

  size_t argi = 0;
  kern_set_arg(kernel, inp_buf);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &stride);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &filter_width);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &filter_height);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &input->cols);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &input->rows);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &input->dim3);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &type);

  kern_set_size_arg(kernel, sizeof(cl_ulong), &res_width);
  kern_set_size_arg(kernel, sizeof(cl_ulong), &res_height);

  kern_set_arg(kernel, res_buf);

  if (ret) return ret;
  
  size_t ldim1 = 8, ldim2 = 16, ldim3 = 1;

  size_t dim1 = align(xdim, ldim1);
  size_t dim2 = align(ydim, ldim2);
  size_t dim3 = align(zdim, ldim3);
 
  auto glob_range = NDRange(dim1, dim2, dim3);
  auto loc_range = NDRange(ldim1, ldim2, ldim3);

  ret = queue.enqueueNDRangeKernel(kernel, NullRange, glob_range, loc_range);
  if (ret) return ret;

  ret = queue.enqueueReadBuffer(res_buf, CL_TRUE, 0, res_mat_size, res->matrix);

  // printf("POoling res\n");

  // mat_print(res);
  // printf("POoling res\n\n");
  return ret;
}

