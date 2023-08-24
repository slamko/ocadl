float sigmoid(float x) {
    float val = (1.0 / (1.0 + exp(-x)));

    if (isnan(val)) {
        if (!isnan(x)) {
            printf("Sigmoid nan: %f\n", x);
        }
    }
    
    return val; 
}

float sigmoid_deriv(float act) {
    return (act * (1.0 - act));
}

__kernel void dense_ff(__global const float *input,
                       __global const float *weight_mat,
                       __global const float *bias_mat,
                       __global float *res,
                       unsigned long dim) {

    size_t x = get_global_id(0);
    size_t width = get_global_size(0);

    float sum = 0.0;

    for (unsigned long i = 0; i < dim; i++) {
         sum += input[i] * weight_mat[x + i * width];
    }

    float r = sigmoid(sum + bias_mat[x]);
    res[x] = r;
}

__kernel void dense_bp(__global const float *weight_mat,
                       __global const float *prev_act_mat,
                       __global const float *act_mat,
                       __global const float *diff_mat,
                       unsigned long dim,
                       __global float *cache,
                       __global float *wmat_grad,
                       __global float *bmat_grad) {

    size_t x = get_global_id(0);
    size_t width = get_global_size(0);

    float diff = diff_mat[x];
    float cur_act = act_mat[x];
    float cur_act_deriv = sigmoid_deriv(cur_act);
     
    if (isnan(diff)) {
        printf("Nan diff: %f\n", diff);
    } if (isnan(cur_act)) {
        printf("Nan act: %f\n", diff);
    } if (isnan(cur_act_deriv) && !isnan(cur_act)) {
        printf("Fuck : %f\n", 4.0);
    }

    for (unsigned long i = 0; i < dim; i++) {
         size_t wmati = i * width + x;

         float weight = weight_mat[wmati];
         float prev_act = prev_act_mat[i];

         float dprev = 2.0 * diff * cur_act_deriv * weight;
         float dw = 2.0 *diff * cur_act_deriv * prev_act;

         wmat_grad[wmati] = dw;
         cache[wmati] = dprev;
    }

   bmat_grad[x] = 2.0 * diff * cur_act_deriv;
}

