#define CL_TARGET_OPENCL_VERSION 220

#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>
#include <time.h>
#include <immintrin.h>
#include <xmmintrin.h>

/* #define DEBUG */

#include "blac.h"

#ifdef DEBUG 
#define VEC_DIM1 0x4
#else
#define VEC_DIM1 0x401
#endif
#define VEC_LEN (VEC_DIM1 * VEC_DIM1)
#define VEC_SIZE (VEC_LEN * sizeof(float))

size_t mat_mem_size(const struct mat *mat) {
    return (mat->rows * mat->cols * mat->dim3 * sizeof(*mat->matrix));
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

    return mat3_of_array(matrix, rows, cols, dim3);
}

struct mat mat_make(size_t rows, size_t cols) {
    return mat3_nil(rows, cols, 1);
}

struct mat mat3_nil(size_t rows, size_t cols, size_t dim3) {
    size_t mat_dims = rows * cols;
    
    float *matrix = calloc(mat_dims, sizeof *matrix);

    return mat3_of_array(matrix, rows, cols, dim3);
}

struct mat mat_nil(size_t rows, size_t cols) {
    return mat3_nil(rows, cols, 1);
}

struct mat mat3_random(size_t rows, size_t cols, size_t dim3) {
    struct mat mat = mat3_make(rows, cols, dim3);

    for (size_t i = 0; i < rows * cols; i++) {
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

int mat_gemm(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *restrict a, const struct mat *restrict b,
             struct mat *c) {

    cl_int ret = {0};

    printf ("Cols : %zu, Rows : %zu\n", a->cols, b->rows);
    if (a->cols != b->rows) {
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

    cl_kernel kernel = clCreateKernel(program, "matrix_mul", &ret);
    if (ret) {
        goto cleanup;
    }

    cl_ulong mul_dim = a->cols;
    ret = clSetKernelArg(kernel, 0, sizeof a_mem, &a_mem);
    ret = clSetKernelArg(kernel, 1, sizeof b_mem, &b_mem);
    ret = clSetKernelArg(kernel, 2, sizeof c_mem, &c_mem);
    ret = clSetKernelArg(kernel, 3, sizeof mul_dim, &mul_dim); 

    if (ret) goto cleanup;

    size_t global_work_size [2] = { b->cols, a->rows };
    size_t dim1 = (b->cols < 32) ? b->cols : 32;
    size_t dim2 = (a->rows < 32) ? a->rows : 32;
    
    size_t local_work_size [2] = { dim1, dim2};

    ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

    ret = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0u, c_mat_size,
                              c->matrix, 0, NULL, NULL);
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
        goto cleanup;
    }

    ret = clSetKernelArg(kernel, 0, sizeof a_mem, &a_mem);
    ret = clSetKernelArg(kernel, 1, sizeof b_mem, &b_mem);
    ret = clSetKernelArg(kernel, 2, sizeof c_mem, &c_mem);
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
    ret = clReleaseKernel(kernel);

  cleanup:
    clReleaseMemObject(c_mem);
  clean_b_mem:
    clReleaseMemObject(b_mem);
  clean_a_mem:
    clReleaseMemObject(a_mem);
    return ret;
}

int mat_sum(cl_context context, cl_command_queue queue, cl_program program,
            const struct mat *mat, float *res) {
    cl_int ret = {0};

    if (mat->cols == 0 || mat->rows == 0) {
        *res = 0.0;
        return 0;
    }
    
    size_t mat_size = mat_mem_size(mat);

    cl_mem mat_mem = clCreateBuffer(context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  mat_size, mat->matrix, &ret);
    if (ret) {
        return ret;
    }

    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  sizeof *res, NULL, &ret);
    if (ret) {
        goto clean_mat_mem;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_sum", &ret);
    if (ret) {
        goto cleanup;
    }

    ret = clSetKernelArg(kernel, 0, sizeof mat_mem, &mat_mem);
    ret = clSetKernelArg(kernel, 2, sizeof c_mem, &c_mem);
    if (ret) goto cleanup;

    size_t global_work_size [2] = { mat->rows, mat->cols };
    size_t dim1 = (mat->rows < 32) ? mat->rows : 32;
    size_t dim2 = (mat->cols < 32) ? mat->cols : 32;
    
    size_t local_work_size [2] = { dim1, dim2};

    ret = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, /*  */
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

    ret = clEnqueueReadBuffer(queue, c_mem, CL_TRUE, 0u, sizeof *res,
                              res, 0, NULL, NULL);
    if (ret) goto cleanup;
    
    clFlush(queue);

  cleanup:
    clReleaseMemObject(c_mem);
  clean_mat_mem:
    clReleaseMemObject(mat_mem);
    return ret;
}

int mat_scale(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *mat, float scale) {

    cl_int ret = {0};

    if (mat->cols == 0 || mat->rows == 0 || mat->dim3 == 0) {
        return 1;
    }
    
    size_t mat_size = mat_mem_size(mat);

    cl_mem mat_mem = clCreateBuffer(context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  mat_size, mat->matrix, &ret);
    if (ret) {
        return ret;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_scale", &ret);
    if (ret) {
        goto cleanup;
    }

    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_mem);
    ret = clSetKernelArg(kernel, 1, sizeof(scale), &scale);

    if (ret) goto cleanup;

    size_t global_work_size [3] = {  mat->rows, mat->cols, mat->dim3 };
    size_t dim1 = (mat->rows < 16) ? mat->rows : 16;
    size_t dim2 = (mat->cols < 16) ? mat->cols : 16;
    size_t dim3 = (mat->dim3 < 16) ? mat->dim3 : 16;
    
    size_t local_work_size [3] = { dim1, dim2, dim3};

    ret = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size,
                                 local_work_size, 0, NULL, NULL);
    if (ret) goto cleanup;

    clFlush(queue);

    ret = clReleaseKernel(kernel);
  cleanup:
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
        goto cleanup;
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
    clReleaseMemObject(c_mem);
  clean_b_mem:
    clReleaseMemObject(b_mem);
  clean_a_mem:
    clReleaseMemObject(a_mem);
    return ret;
}

struct mat mat_mult(cl_context context, cl_command_queue queue,
                    cl_program program, struct mat *restrict a,
                    struct mat *restrict b) {

    struct mat res_mul = mat_make(a->rows, b->cols);
    int res = mat_gemm(context, queue, program, a, b, &res_mul);

    if (res) {
        fprintf(stderr, "Multiplication failed: %d\n", res);
        exit(1);
    }

    return res_mul;
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

int prog () {
    cl_int ret = {0};
    cl_command_queue command_queue;
    cl_context context;
    cl_device_id device_id;
    cl_program program;

    ret = ocl_init(&command_queue, &context, &device_id);
    if (ret) return ret;
    ret = load_program("add.c", &program, context, &device_id);
    if (ret) return ret;

    /* size_t max = 0; */
    /* clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (size_t), */
                    /* &max, NULL);  */
    /* printf("max %zu\n", max); */

    /* struct mat a = mat_random(VEC_DIM1, VEC_DIM1); */
    /* struct mat b = mat_random(VEC_DIM1, VEC_DIM1); */
    /* struct mat c = mat_nil(VEC_DIM1, VEC_DIM1); */

    struct mat mat = mat_random(VEC_DIM1, VEC_DIM1);
    struct mat kernel = mat_random(2, 2);
    struct mat res = mat_nil(VEC_DIM1 - 1, VEC_DIM1 - 1);

    clock_t start = clock();

    /* ret = mat_gemm(context, command_queue, program, &a, &b, &c); */
    /* ret = cpu_gemm(&a, &b, &c); */
    /* ret = simd_gemm(&a, &b, &c); */

    ret = mat_convolve(context, command_queue, program, &mat, &kernel, &res);
    clock_t end = clock();

    float time_el = (float)((end - start) * 1000 * 1000) / CLOCKS_PER_SEC;

    printf ("Convolution time: %f us. Status: %d\n", time_el, ret);

#ifdef DEBUG
    for(int i = 0; i < VEC_LEN; i++) {
        if (i % mat.rows == 0) {
            putc('\n', stdout);
        }

        printf("%f  ", mat.matrix[i]);

    }

    putc('\n', stdout);

    for(int i = 0; i < kernel.rows * kernel.cols; i++) {
        if (i % kernel.rows == 0) {
            putc('\n', stdout);
        }

        printf("%f  ", kernel.matrix[i]);

    }

    putc('\n', stdout);

    for(int i = 0; i < res.rows * res.cols; i++) {
        if (i % res.rows == 0) {
            putc('\n', stdout);
        }

        printf("%f  ", res.matrix[i]);

    }

    putc('\n', stdout);



    /* ret = mat_gemm(context, command_queue, program, &a, &b, &c); */

    /* for(int i = 0; i < VEC_LEN; i++) */
        /* printf("%f * %f = %f\n", a.matrix[i], b.matrix[i], c.matrix[i]); */
#endif

    return 0;
}

int example() {
    float *a = malloc(VEC_SIZE);
    float *b = malloc(VEC_SIZE);

    for (int i = 0; i < VEC_LEN; i++) {
        a[i] = (float)i;
        b[i] = (float)i;
    }

    FILE *fp = fopen ("add.c", "r");

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

    cl_platform_id platform_id = {0};
    cl_device_id device_id = {0};   
    cl_uint ret_num_devices = {0};
    cl_uint ret_num_platforms = {0};
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    if (ret) {
        fprintf(stderr, "No CL platforms found\n");
        exit(1);
    }

    printf("CL platforms number: %d\n", ret_num_platforms);

    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, 
            &device_id, &ret_num_devices);

    if (ret) {
        fprintf(stderr, "No CL devices found %d\n", ret);
        exit(1);
    }
 
    // Create an OpenCL context
    cl_context context =
        clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    if (ret) {
        fprintf(stderr, "Failed to instantiate the context, error = %d\n",
                ret);
        exit(1);
    }
 
    // Create a command queue
    cl_command_queue command_queue =
        clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);

    if (ret) {
        fprintf(stderr, "Failed to instantiate the command queue\n");
        exit(1);
    }

    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
            VEC_SIZE, NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            VEC_SIZE, NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
            VEC_SIZE, NULL, &ret);
 // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
            VEC_LEN * sizeof(int), a, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
            VEC_LEN * sizeof(int), b, 0, NULL, NULL);
 
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&src, (const size_t *)&src_len, &ret);
 
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matrix_mul", &ret);
    cl_ulong mat_dim = VEC_DIM1;
 
    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem_obj);
    ret = clSetKernelArg(kernel, 3, sizeof(mat_dim), &mat_dim);
 
    // Execute the OpenCL kernel on the list
    size_t global_item_size[2] = {VEC_DIM1, VEC_DIM1};
    size_t local_item_size[2]  = {VEC_DIM1, VEC_DIM1} ; 

    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, 
            global_item_size, local_item_size, 0, NULL, NULL);

    if (ret) {
        printf("Kernel execution failed %d\n", ret);
        exit(1);
    }
 
    // Read the memory buffer C on the device to the local variable C
    float *c = malloc(VEC_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
            VEC_SIZE, c, 0, NULL, NULL);
 
    for(int i = 0; i < VEC_LEN; i++)
        printf("%f * %f = %f\n", a[i], b[i], c[i]);
 
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(a);
    free(b);
    free(c);
    
    return 0;
}
