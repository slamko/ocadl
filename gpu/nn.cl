float sigmoid(float x) {
    float val = (1.0 / (1.0 + exp(-x)));
   
    return val; 
}

float sigmoid_deriv(float act) {
    return (act * (1.0 - act));
}

__kernel void dense_ff(__global __read_only const float *input,
                       __global __read_only const float *weight_mat,
                       __global __read_only const float *bias_mat,
                       __global __write_only float *res,
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

__kernel void dense_bp(__global __read_only const float *weight_mat,
                       __global __read_only const float *prev_act_mat,
                       __global __read_only const float *act_mat,
                       __global __read_only const float *diff_mat,
                       unsigned long dim,
                       __global __write_only float *cache,
                       __global __write_only float *wmat_grad,
                       __global __write_only float *bmat_grad) {

    size_t x = get_global_id(0);
    size_t width = get_global_size(0);

    float diff = diff_mat[x];
    float cur_act = act_mat[x];
    float cur_act_deriv = sigmoid_deriv(cur_act);
     
    for (unsigned long i = 0; i < dim; i++) {
         size_t wmati = i * width + x;

         float weight = weight_mat[wmati];
         float prev_act = prev_act_mat[i];

         float dprev = diff * cur_act_deriv * weight;
         float dw = diff * cur_act_deriv * prev_act;

         wmat_grad[wmati] = dw;
         cache[wmati] = dprev;
    }

   bmat_grad[x] = diff * cur_act_deriv;
}

