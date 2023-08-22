#include <CL/cl.h>
#include "blac.h"

int fully_connected_bp(cl_context context, cl_command_queue queue,
                       cl_program program, struct mat *input,
                       struct mat *weight_mat, struct mat *bias_vec,
                       struct mat *res);


int fully_connected_ff(cl_context context, cl_command_queue queue,
                       cl_program program, struct mat *input,
                       struct mat *weight_mat, struct mat *bias_vec,
                       struct mat *res);

