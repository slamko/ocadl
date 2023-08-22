#include <CL/cl.h>
#include "blac.h"

int fully_connected_bp(cl_context context, cl_command_queue queue,
                       cl_program program,
                       struct mat *weight_mat,
                       struct mat *prev_act_vec,
                       struct mat *act_mat,
                       struct mat *diff_mat,
                       struct mat *prev_diff,
                       struct mat *wmat_grad,
                       struct mat *bgrad_mat);


int fully_connected_ff(cl_context context, cl_command_queue queue,
                       cl_program program, struct mat *input,
                       struct mat *weight_mat, struct mat *bias_vec,
                       struct mat *res);

