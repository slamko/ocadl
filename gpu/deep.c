#define CL_TARGET_OPENCL_VERSION 220

#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>
#include <time.h>
#include <immintrin.h>
#include <xmmintrin.h>

/* #define DEBUG */

#include "deep.h"

int fully_connected_bp(cl_context context, cl_command_queue queue,
                       cl_program program,
                       struct mat *weight_mat,
                       struct mat *prev_act_vec,
                       struct mat *act_vec,
                       struct mat *diff_vec,
                       struct mat *prev_diff_vec,
                       struct mat *wgrad_mat,
                       struct mat *bgrad_vec) {

    cl_int ret = {0};

    if (act_vec->cols != weight_mat->cols
        || weight_mat->rows != prev_act_vec->cols) {
        return 1;
    }
    
    *prev_diff_vec = mat_nil(1, weight_mat->rows);
    *wgrad_mat = mat_nil(weight_mat->rows, weight_mat->cols);
    *bgrad_vec = mat_nil(1, act_vec->cols);

    size_t prev_act_size = mat_mem_size(prev_act_vec);
    size_t act_size = mat_mem_size(act_vec);
    size_t diff_size = mat_mem_size(diff_vec);
    size_t wmat_size = mat_mem_size(weight_mat);

    size_t prev_diff_size = mat_mem_size(prev_diff_vec);
    size_t wgrad_size = mat_mem_size(wgrad_mat);
    size_t bgrad_size = mat_mem_size(bgrad_vec);


    cl_mem wmat_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  wmat_size, weight_mat->matrix, &ret);
    if (ret) {
        return ret;
    }

    cl_mem prev_act_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  prev_act_size, prev_act_vec->matrix, &ret);
    if (ret) {
        goto clean_wmat_mem;
    }

    cl_mem act_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  act_size, act_vec->matrix, &ret);
    if (ret) {
        goto clean_prev_act_mem;
    }

    cl_mem diff_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  diff_size, diff_vec->matrix, &ret);
    if (ret) {
        goto clean_act_mem;
    }


    cl_mem prev_diff_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  prev_diff_size, NULL, &ret);
    if (ret) {
        goto clean_diff_mem;
    }

    cl_mem wgrad_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  wgrad_size, NULL, &ret);
    if (ret) {
        goto clean_prev_diff_mem;
    }

    cl_mem bgrad_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  bgrad_size, NULL, &ret);
    if (ret) {
        goto clean_wgrad_mem;
    }


    cl_kernel kernel = clCreateKernel(program, "dense_bp", &ret);
    if (ret) {
        goto clean_bgrad_mem;
    }

    cl_ulong dim = weight_mat->rows;

    struct mat cache_vec = mat_nil(dim, act_vec->cols);
    cl_mem cache_mem = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY,
                                      mat_mem_size(&cache_vec),
                                      NULL, &ret);
    if (ret) {
        goto clean_cache;
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &wmat_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &prev_act_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &act_mem);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &diff_mem);
    ret = clSetKernelArg(kernel, 4, sizeof(dim), &dim);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &cache_mem);
    /* ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), &prev_diff_mem); */
    ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), &wgrad_mem);
    ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), &bgrad_mem);
    if (ret) goto clean_cache_mem;

    size_t width = weight_mat->cols;
    size_t global_work_size [1] = { width };
    size_t dim1 = (width < 32) ? width : 32;
    
    size_t local_work_size [1] = { dim1 };

    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto clean_cache_mem;

    ret = clEnqueueReadBuffer(queue, cache_mem, CL_TRUE, 0u,
                              mat_mem_size(&cache_vec), cache_vec.matrix,
                              0, NULL, NULL);
    if (ret) goto cleanup;

    for (size_t i = 0; i < dim; i++) {
        prev_diff_vec->matrix[i] = 0.0;
        for (size_t j = 0; j < cache_vec.cols; j++) {
            prev_diff_vec->matrix[i] +=
                cache_vec.matrix[i * cache_vec.cols + j];
        }
    }
 
    ret = clEnqueueReadBuffer(queue, wgrad_mem, CL_TRUE, 0u, wgrad_size,
                              wgrad_mat->matrix, 0, NULL, NULL);
    if (ret) goto cleanup;
 
    ret = clEnqueueReadBuffer(queue, bgrad_mem, CL_TRUE, 0u, bgrad_size,
                              bgrad_vec->matrix, 0, NULL, NULL);
    if (ret) goto cleanup;

    /*
    printf("\nw diff\n");
    mat_print(stdout, diff_vec);
    printf("\nw grad\n");
    mat_print(stdout, wgrad_mat);
    printf("\nB grad\n");
    mat_print(stdout, bgrad_vec);
    */
    
    clFlush(queue);

  cleanup:
    ret = clReleaseKernel(kernel);
  clean_cache_mem:
    clReleaseMemObject(cache_mem);
  clean_cache:
    mat_free(&cache_vec);
  clean_bgrad_mem:
    clReleaseMemObject(bgrad_mem);
  clean_wgrad_mem:
    clReleaseMemObject(wgrad_mem);
  clean_prev_diff_mem:
    clReleaseMemObject(prev_diff_mem);
  clean_diff_mem:
    clReleaseMemObject(diff_mem);
  clean_act_mem:
    clReleaseMemObject(act_mem);
  clean_prev_act_mem:
    clReleaseMemObject(prev_act_mem);
  clean_wmat_mem:
    clReleaseMemObject(wmat_mem);
    return ret;
}

int fully_connected_ff(cl_context context, cl_command_queue queue,
                       cl_program program, struct mat *input,
                       struct mat *weight_mat, struct mat *bias_vec,
                       struct mat *res) {

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

    cl_mem inp_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  inp_mat_size, input->matrix, &ret);
    if (ret) {
        return ret;
    }

    cl_mem wmat_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  wmat_size, weight_mat->matrix, &ret);
    if (ret) {
        goto clean_inp_mem;
    }

    cl_mem bmat_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bmat_size, bias_vec->matrix, &ret);
    if (ret) {
        goto clean_wmat_mem;
    }

    cl_mem res_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  res_mat_size, NULL, &ret);
    if (ret) {
        goto clean_bmat_mem;
    }

    cl_kernel kernel = clCreateKernel(program, "dense_ff", &ret);
    if (ret) {
        goto clean_res_mem;
    }

    cl_ulong mat_dim = weight_mat->rows;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inp_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &wmat_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bmat_mem);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &res_mem);
    ret = clSetKernelArg(kernel, 4, sizeof(mat_dim), &mat_dim);
    if (ret) goto clean_res_mem;

    cl_ulong dim = weight_mat->cols;
    size_t global_work_size [1] = { dim };
    size_t dim1 = (dim < 32) ? dim : 32;
    
    size_t local_work_size [1] = { dim1 };

    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto clean_res_mem;

    ret = clEnqueueReadBuffer(queue, res_mem, CL_TRUE, 0u, res_mat_size,
                              res->matrix, 0, NULL, NULL);

    if (ret) goto cleanup;
    
    clFlush(queue);

  cleanup:
    ret = clReleaseKernel(kernel);
  clean_res_mem:
    clReleaseMemObject(res_mem);
  clean_wmat_mem:
    clReleaseMemObject(wmat_mem);
  clean_bmat_mem:
    clReleaseMemObject(bmat_mem);
  clean_inp_mem:
    clReleaseMemObject(inp_mem);
    return ret;
}

