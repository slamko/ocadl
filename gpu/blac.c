#define CL_TARGET_OPENCL_VERSION 220

#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>
#include <time.h>
#include <immintrin.h>
#include <xmmintrin.h>

/* #define DEBUG */

#include "blac.h"

size_t mat_mem_size(const struct mat *mat) {
    return (mat->rows * mat->cols * mat->dim3 * sizeof(*mat->matrix));
}

void mat_free(struct mat *mat) {
    free(mat->matrix);
}

struct mat mat3_of_array(float *matrix, size_t rows, size_t cols, size_t dim3) {
    struct mat mat = {
        .rows = rows,
        .cols = cols,
        .dim3 = dim3,
    };

    size_t mat_dims = rows * cols * dim3;
    
    mat.matrix = matrix;
    
    mat.stride = cols;
    return mat;
}

struct mat mat_of_array(float *arr, size_t rows, size_t cols) {
    return mat3_of_array(arr, rows, cols, 1);
}

struct mat mat3_make(size_t rows, size_t cols, size_t dim3) {
    size_t mat_dims = rows * cols;
    
    float *matrix = malloc(mat_dims * sizeof(*matrix));

    if (!matrix) {
        fatal_error("Matrix allocation failed\n");
    }

    return mat3_of_array(matrix, rows, cols, dim3);
}

struct mat mat_make(size_t rows, size_t cols) {
    return mat3_nil(rows, cols, 1);
}

struct mat mat3_nil(size_t rows, size_t cols, size_t dim3) {
    size_t mat_dims = rows * cols;
    
    float *matrix = calloc(mat_dims, sizeof *matrix);
    if (!matrix) {
        fatal_error("Matrix allocation failed\n");
    }

    return mat3_of_array(matrix, rows, cols, dim3);
}

struct mat mat_nil(size_t rows, size_t cols) {
    return mat3_nil(rows, cols, 1);
}

struct mat mat3_random(size_t rows, size_t cols, size_t dim3) {
    struct mat mat = mat3_make(rows, cols, dim3);

    for (size_t i = 0; i < rows * cols * dim3; i++) {
        mat.matrix[i] = ((float)(rand() % 1000) / 500.) - 1.0f;
    }
    
    return mat;
}

struct mat mat_random(size_t rows, size_t cols) {
    return mat3_random(rows, cols, 1);
}

void mat_print(FILE *out, struct mat *mat) {
    for (size_t d3 = 0; d3 < mat->dim3; d3++) {
        for (size_t r = 0; r < mat->rows; r++) {
            for (size_t c = 0; c < mat->cols; c++) {
                fprintf(out, "%f   ",
                       mat->matrix[c + r * mat->cols + d3 * mat->rows]);
            }
            fputc('\n', out);
        }

        fputs("\n\n", out);
    }
}

int mat_convolve(cl_context context, cl_command_queue queue,
                 cl_program program, const struct mat *restrict mat,
                 const struct mat *restrict conv_kern, struct mat *res) {
    cl_int ret = {0};

    if (mat->cols < conv_kern->cols || mat->rows < conv_kern->rows) {
        return 1;
    }
    
    size_t mat_size = mat_mem_size(mat);
    size_t kernel_size = mat_mem_size(conv_kern);
    size_t res_mat_size = mat_mem_size(res);

    cl_mem a_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  mat_size, mat->matrix, &ret);
    if (ret) {
        return ret;
    }

    cl_mem b_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  kernel_size, conv_kern->matrix, &ret);
    if (ret) {
        goto clean_a_mem;
    }

    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  res_mat_size, NULL, &ret);
    if (ret) {
        goto clean_b_mem;
    }

    cl_kernel kernel = clCreateKernel(program, "convolve", &ret);
    if (ret) {
        goto cleanup;
    }

    cl_ulong mat_width = mat->cols;
    cl_ulong res_width = res->cols;
    cl_ulong kernel_width = conv_kern->cols;
    cl_ulong kernel_height = conv_kern->rows;

    ret = clSetKernelArg(kernel, 0, sizeof a_mem, &a_mem);
    ret = clSetKernelArg(kernel, 1, sizeof b_mem, &b_mem);
    ret = clSetKernelArg(kernel, 2, sizeof c_mem, &c_mem);
    ret = clSetKernelArg(kernel, 3, sizeof (cl_ulong), &mat_width); 
    ret = clSetKernelArg(kernel, 4, sizeof (cl_ulong), &res_width); 
    ret = clSetKernelArg(kernel, 5, sizeof (cl_ulong), &kernel_width); 
    ret = clSetKernelArg(kernel, 6, sizeof (cl_ulong), &kernel_height); 

    if (ret) goto cleanup;

    size_t global_work_size [2] = { res->cols, res->rows };
    size_t local_work_size [2] = { 32, 32 };

    ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

    ret = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0u, res_mat_size,
                              res->matrix, 0, NULL, NULL);
    if (ret) goto cleanup;
    
    clFlush(queue);

  cleanup:
    clReleaseMemObject(c_mem);
  clean_b_mem:
    clReleaseMemObject(b_mem);
  clean_a_mem:
    clReleaseMemObject(a_mem);
    return ret;
}

int vec_sum(cl_context context, cl_command_queue queue, cl_program program,
            const struct mat *vec, float *res) {
    *res = 0.0;

    for(size_t i = 0; i < vec->cols; i++) {
        *res += vec->matrix[i];
    }

    return 0;
    
    cl_int ret = {0};

    if (vec->cols == 0 || vec->rows == 0) {
        return 0;
    }
    
    size_t mat_size = mat_mem_size(vec);

    cl_mem mat_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  mat_size, vec->matrix, &ret);
    if (ret) {
        return ret;
    }

    cl_mem res_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  sizeof *res, NULL, &ret);
    if (ret) {
        goto clean_mat_mem;
    }

    cl_kernel kernel = clCreateKernel(program, "vector_sum", &ret);
    if (ret) {
        goto cleanup;
    }

    ret = clSetKernelArg(kernel, 0, sizeof (cl_mem), &mat_mem);
    ret = clSetKernelArg(kernel, 1, sizeof (cl_mem), &res_mem);
    if (ret) goto cleanup;

    size_t global_work_size [1] = { vec->cols };
    size_t dim1 = (vec->cols < 32) ? vec->cols : 32;
    
    size_t local_work_size [1] = { dim1};

    ret = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

    ret = clEnqueueReadBuffer(queue, res_mem, CL_TRUE, 0u, sizeof *res,
                              res, 0, NULL, NULL);
    if (ret) goto cleanup;
    
    clFlush(queue);

  cleanup:
    clReleaseMemObject(res_mem);
  clean_mat_mem:
    clReleaseMemObject(mat_mem);
    return ret;
}

int mat_sub(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *a, const struct mat *b, struct mat *c) {

    cl_int ret = {0};

    if (a->cols != b->cols || a->rows != b->rows) {
        return 1;
    }
    
    size_t a_mat_size = mat_mem_size(a);
    size_t b_mat_size = mat_mem_size(b);
    size_t c_mat_size = mat_mem_size(c);

    cl_mem a_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  a_mat_size, a->matrix, &ret);
    if (ret) {
        return ret;
    }

    cl_mem b_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  b_mat_size, b->matrix, &ret);
    if (ret) {
        goto clean_a_mem;
    }

    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  c_mat_size, NULL, &ret);
    if (ret) {
        goto clean_b_mem;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_sub", &ret);
    if (ret) {
        goto clean_c_mem;
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    if (ret) goto cleanup;

    size_t global_work_size [2] = { a->rows, a->cols };
    size_t dim1 = (a->rows < 16) ? a->rows : 16;
    size_t dim2 = (a->cols < 16) ? a->cols : 16;

    size_t local_work_size [2] = { dim1, dim2 };

    ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

    ret = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0u, c_mat_size,
                              c->matrix, 0, NULL, NULL);
    if (ret) goto cleanup;
    
    clFlush(queue);

  cleanup:
    clReleaseKernel(kernel);
  clean_c_mem:
    clReleaseMemObject(c_mem);
  clean_b_mem:
    clReleaseMemObject(b_mem);
  clean_a_mem:
    clReleaseMemObject(a_mem);

    return ret;
}

int mat_scale(cl_context context, cl_command_queue queue, cl_program program,
              const struct mat *mat, struct mat *res, float scale) {

    cl_int ret = {0};

    if (mat->cols == 0 || mat->rows == 0 || mat->dim3 == 0) {
        return 1;
    }
    
    *res = mat_make(mat->rows, mat->cols);
    size_t mat_size = mat_mem_size(mat);
    size_t res_size = mat_mem_size(res);

    cl_mem mat_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  mat_size, mat->matrix, &ret);
    if (ret) {
        return ret;
    }
    cl_mem res_mem = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY,
                                  res_size, NULL, &ret);
    if (ret) {
        goto clean_mat_mem;
    }


    cl_kernel kernel = clCreateKernel(program, "matrix_scale", &ret);
    if (ret) {
        goto clean_res_mem;
    }

    cl_float cl_scale = scale;
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_mem);
    if (ret) goto cleanup;
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &res_mem);
    if (ret) goto cleanup;
    ret = clSetKernelArg(kernel, 2, sizeof(cl_float), &cl_scale);
    if (ret) goto cleanup;

    size_t global_work_size [3] = {  mat->rows, mat->cols, mat->dim3 };
    size_t dim1 = (mat->rows < 16) ? mat->rows : 16;
    size_t dim2 = (mat->cols < 16) ? mat->cols : 16;
    size_t dim3 = (mat->dim3 < 16) ? mat->dim3 : 16;
    
    size_t local_work_size [3] = { dim1, dim2, dim3};

    ret = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

    ret = clEnqueueReadBuffer(queue, res_mem, CL_TRUE, 0u, res_size,
                              res->matrix, 0, NULL, NULL);
    if (ret) goto cleanup;
 
    clFlush(queue);

  cleanup:
    clReleaseKernel(kernel);
  clean_res_mem:
    clReleaseMemObject(res_mem);
  clean_mat_mem:
    clReleaseMemObject(mat_mem);
    return ret;
}

int mat_add(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *a, const struct mat *b,
             struct mat *c) {

    cl_int ret = {0};

    if (a->cols != b->cols || a->rows != b->rows) {
        return 1;
    }
    
    size_t a_mat_size = mat_mem_size(a);
    size_t b_mat_size = mat_mem_size(b);
    size_t c_mat_size = mat_mem_size(c);

    cl_mem a_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  a_mat_size, a->matrix, &ret);
    if (ret) {
        return ret;
    }

    cl_mem b_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  b_mat_size, b->matrix, &ret);
    if (ret) {
        goto clean_a_mem;
    }

    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  c_mat_size, NULL, &ret);
    if (ret) {
        goto clean_b_mem;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_add", &ret);
    if (ret) {
        goto clean_c_mem;
    }

    ret = clSetKernelArg(kernel, 0, sizeof (cl_mem), &a_mem);
    ret = clSetKernelArg(kernel, 1, sizeof (cl_mem), &b_mem);
    ret = clSetKernelArg(kernel, 2, sizeof (cl_mem), &c_mem);
    if (ret) goto cleanup;

    size_t global_work_size [2] = { a->rows, a->cols };
    size_t dim1 = (a->rows < 16) ? a->rows : 16;
    size_t dim2 = (b->cols < 16) ? b->cols : 16;
    
    size_t local_work_size [2] = { dim1, dim2};

    ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

    ret = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0u, c_mat_size,
                              c->matrix, 0, NULL, NULL);
    if (ret) goto cleanup;
    
    clFlush(queue);

  cleanup:
    clReleaseKernel(kernel);
  clean_c_mem:
    clReleaseMemObject(c_mem);
  clean_b_mem:
    clReleaseMemObject(b_mem);
  clean_a_mem:
    clReleaseMemObject(a_mem);
    return ret;
}

int load_program(const char *prog_name,
                 cl_program *program, cl_context context, cl_device_id *dev_ids) {

    cl_int ret = {0};
    FILE *fp = fopen (prog_name, "r");

    if (!fp) {
        fprintf(stderr, "No source file found");
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    size_t src_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *src = malloc(src_len + 1);
    if (fread(src, sizeof *src, src_len, fp) != src_len) {
        fprintf(stderr, "Read failed");
        exit(1);
    }
    fclose(fp);

    *program = clCreateProgramWithSource(context, 1, 
            (const char **)&src, (const size_t *)&src_len, &ret);
 
    // Build the program
    ret = clBuildProgram(*program, 1, dev_ids, NULL, NULL, NULL);
    return ret;
}

int ocl_init(cl_command_queue *command_queue, cl_context *context,
             cl_device_id *device_id) {

    cl_platform_id platform_id = {0};
    cl_uint ret_num_devices = {0};
    cl_uint ret_num_platforms = {0};
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    if (ret) {
        fprintf(stderr, "No CL platforms found\n");
        exit(1);
    }

    printf("CL platforms number: %d\n", ret_num_platforms);

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, 
            device_id, &ret_num_devices);

    if (ret) {
        fprintf(stderr, "No CL devices found %d\n", ret);
        exit(1);
    }
 
    // Create an OpenCL context
    *context =
        clCreateContext(NULL, 1, device_id, NULL, NULL, &ret);

    if (ret) {
        fprintf(stderr, "Failed to instantiate the context, error = %d\n",
                ret);
        exit(1);
    }
 
    // Create a command queue
    *command_queue =
        clCreateCommandQueueWithProperties(*context, *device_id, NULL, &ret);

    if (ret) {
        fprintf(stderr, "Failed to instantiate the command queue\n");
        exit(1);
    }

    srand(time(NULL));
    return ret;
}

void ocl_finish(cl_context context, cl_command_queue queue,
                cl_program *progs, size_t prog_num) {
    clFinish(queue);

    for (size_t i = 0; i < prog_num; i++) {
        clReleaseProgram(progs[i]);
    }

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
