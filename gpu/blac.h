#include <CL/cl.h>
#include <stddef.h>

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

struct mat mat3_of_array(float *matrix,
                         size_t rows, size_t cols, size_t dim3);

struct mat mat3_nil(size_t rows, size_t cols, size_t dim3);

struct mat mat3_random(size_t rows, size_t cols, size_t dim3);

struct mat mat_nil(size_t rows, size_t cols);

struct mat mat_of_array(float *arr, size_t rows, size_t cols);

int fully_connected_ff(cl_context context, cl_command_queue queue,
                       cl_program program, struct mat *input,
                       struct mat *weight_mat, struct mat *bias_vec,
                       struct mat *res);

int mat_add(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *a, const struct mat *b,
            struct mat *c);

int vec_sub(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *a, const struct mat *b,
            struct mat *c);

int vec_sum(cl_context context, cl_command_queue queue, cl_program program,
            const struct mat *vec, float *res);

int mat_sum(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *mat, float *res);

int ocl_init(cl_command_queue *command_queue, cl_context *context,
             cl_device_id *device_id);

int load_program(const char *prog_name, cl_program *program,
                 cl_context context, cl_device_id *dev_ids);

int mat_gemm(cl_context context, cl_command_queue queue, cl_program program,
             const struct mat *a, const struct mat *b, struct mat *c);
