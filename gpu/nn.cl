float sigmoid(float x) {
    return (1.0 / (1.0 + exp(-x))); 
}

float sigmoid_deriv(float act) {
    return (act * (1.0 - act));
}

void atomic_add_f(volatile __global float *addr, float val) {
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr,
        expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
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
                       __global float *prev_diff,
                       __global float *wmat_grad,
                       __global float *bmat_grad) {
    size_t x = get_global_id(0);
    size_t width = get_global_size(0);

    float diff = diff_mat[x];
    float cur_act = act_mat[x];
    float cur_act_deriv = sigmoid_deriv(cur_act);
     
    for (unsigned long i = 0; i < dim; i++) {
         float weight = weight_mat[x + i * width];
         float prev_act = prev_act_mat[i];

         float dprev = 2.0 * diff * cur_act_deriv * weight;
         float dw = 2.0 * diff * cur_act_deriv * prev_act;

         wmat_grad[width * i + x] = dw;
         atomic_add_f(&prev_diff[i], dprev);
    }

   bmat_grad[x] = 2.0 * diff * cur_act_deriv;
}

