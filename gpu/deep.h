#ifndef DEEP_H
#define DEEP_H

#include "blasc.h"
#include <stdbool.h>

int fully_connected_bp(
                       const struct mat *weight_mat,
                       const struct mat *prev_act_vec,
                       const struct mat *act_vec,
                       const struct mat *diff_vec,
                       struct mat *prev_diff_vec,
                       struct mat *wgrad_mat,
                       struct mat *bgrad_mat,
                       long actf,
                       _Bool prev_layer);

int fully_connected_ff(const struct mat *input,
                       const struct mat *weight_mat,
                       const struct mat *bias_vec,
                       struct mat *res,
                       long actf);

int conv_ff(const struct mat *input,
                       const struct mat *kernels,
                       const struct mat *bias_vec,
                       long actf,
                       unsigned long padding,
                       unsigned long stride,
                       unsigned long res_width,
                       unsigned long res_height,
                       struct mat *res);
 
int pooling_ff(const struct mat *input,
                          long type,
                          unsigned long stride,
                          unsigned long res_width,
                          unsigned long res_height,
                          unsigned long filter_width,
                          unsigned long filter_height,
                          struct mat *res);
#endif
