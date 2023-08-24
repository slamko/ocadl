#ifndef BLAC_H
#define BLAC_H

#include <CL/cl.h>
#include <stddef.h>
#include <stdio.h>

struct mat {
    float *matrix;
    size_t rows;
    size_t cols;
    size_t dim3;

    size_t stride;

    size_t start_row;
    size_t start_col;
    size_t start_dim3;
};

#define err(msg) fprintf(stderr, msg)
#define error(msg, ...) fprintf(stderr, msg, __VA_ARGS__)

#define fatal_error(msg)                \
    fprintf(stderr, msg); \
    exit(1);

size_t mat_mem_size(const struct mat *mat);

void mat_free(struct mat *mat);

void mat_print(FILE *out, struct mat *mat);

struct mat mat3_of_array(float *matrix,
                         size_t rows, size_t cols, size_t dim3);

struct mat mat3_nil(size_t rows, size_t cols, size_t dim3);

struct mat mat3_random(size_t rows, size_t cols, size_t dim3);

struct mat mat_nil(size_t rows, size_t cols);

struct mat mat_make(size_t rows, size_t cols);

struct mat mat_of_array(float *arr, size_t rows, size_t cols);


int mat_scale(cl_context context, cl_command_queue queue, cl_program program,
              const struct mat *mat, struct mat *res, float scale);

int mat_add(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *a, const struct mat *b,
            struct mat *c);

int mat_sub(cl_context context, cl_command_queue queue, cl_program program,
            const struct mat *a, const struct mat *b, struct mat *c);

int vec_sum(cl_context context, cl_command_queue queue, cl_program program,
            const struct mat *vec, float *res);

int ocl_init(cl_command_queue *command_queue, cl_context *context,
             cl_device_id *device_id);

void ocl_finish(cl_context context, cl_command_queue queue,
                cl_program *progs, size_t prog_num);

int load_program(const char *prog_name, cl_program *program,
                 cl_context context, cl_device_id *dev_ids);

int mat_gemm(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *a, const struct mat *b, struct mat *c);

#endif
