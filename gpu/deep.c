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
                       cl_program program, struct mat *input,
                       struct mat *weight_mat, struct mat *bias_vec,
                       struct mat *res) {

    cl_int ret = {0};

    if (input->cols != weight_mat->rows
        || weight_mat->cols != bias_vec->cols) {
        return 1;
    }
    
    *res = mat_make(1, weight_mat->cols);
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

    cl_kernel kernel = clCreateKernel(program, "dense_bp", &ret);
    if (ret) {
        goto clean_res_mem;
    }

    cl_ulong dim = weight_mat->cols;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inp_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &wmat_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bmat_mem);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &res_mem);
    ret = clSetKernelArg(kernel, 4, sizeof(dim), &dim);
    if (ret) goto cleanup;

    size_t global_work_size [1] = { dim };
    size_t dim1 = (dim < 32) ? dim : 32;
    
    size_t local_work_size [1] = { dim1 };

    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

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

int fully_connected_ff(cl_context context, cl_command_queue queue,
                       cl_program program, struct mat *input,
                       struct mat *weight_mat, struct mat *bias_vec,
                       struct mat *res) {

    cl_int ret = {0};

    if (input->cols != weight_mat->rows
        || weight_mat->cols != bias_vec->cols) {
        return 1;
    }
    
    *res = mat_make(1, weight_mat->cols);
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

    cl_ulong dim = weight_mat->cols;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inp_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &wmat_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bmat_mem);
    ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &res_mem);
    ret = clSetKernelArg(kernel, 4, sizeof(dim), &dim);
    if (ret) goto cleanup;

    size_t global_work_size [1] = { dim };
    size_t dim1 = (dim < 32) ? dim : 32;
    
    size_t local_work_size [1] = { dim1 };

    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

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

