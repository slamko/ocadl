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
                       int prev_layer);

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

int conv_bp(const struct mat *kernels_mat,
                       const struct mat *prev_act_mat,
                       const struct mat *act_mat,
                       const struct mat *diff_mat,
                       struct mat *prev_diff_mat,
                       struct mat *kern_grad_mat,
                       struct mat *bgrad_vec,
                       long actf,
                       unsigned long stride,
                       unsigned long padding,
                       int prev_layer);

int pooling_bp(
    const struct mat *prev_act_mat,
    const struct mat *diff_mat,
    struct mat *prev_diff_mat,
    long pooling_type,
    unsigned long stride,
    unsigned long padding,
    unsigned long filter_width,
    unsigned long filter_height,
    int prev_layer);


#endif
