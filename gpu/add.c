__kernel void vector_add(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    size_t i = get_global_id(0);

    c[i] = a[i] + b[i];
}

__kernel void vector_sub(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    size_t i = get_global_id(0);

    c[i] = a[i] - b[i];
}

__kernel void vector_sum(__global const int *vec,
                         __global int *c) {
    size_t i = get_global_id(0);

    atomic_add(c, vec[i]);
}

__kernel void matrix_sub(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    size_t x = get_global_id(0);
    size_t glob_size = get_global_size(0);
    size_t y = get_global_id(1);

    c[x + glob_size * y] = a[x + glob_size * y] - b[x + glob_size * y];
}

__kernel void matrix_add(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    size_t x = get_global_id(0);
    size_t glob_size = get_global_size(0);
    size_t y = get_global_id(1);

    c[x + glob_size * y] = a[x + glob_size * y] + b[x + glob_size * y];
}

float sigmoid(float x) {
    return (1.0 / (1.0 + exp(-x))); 
}

float sigmoid_deriv(float act) {
    return act * (1.0 - act);
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
    /* printf("Sigmoid: %f\n", r); */
    res[x] = r;
}

__kernel void dense_bp(__global const float *diff_mat,
                       __global const float *act_mat,
                       __global const float *prev_act_mat,
                       __global const float *weight_mat,
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
         float weight =  weight_mat[x + i * width];
         float prev_act = prev_act_mat[i];

         float dprev = 2.0 * diff * cur_act_deriv * weight;
         float dw = 2.0 * diff * cur_act_deriv * prev_act;

         weight_mat[width * i + x] = dw;
         prev_diff[i] += dprev; // should be atomic
    }

    bmat_grad[x] = 2.0 * diff * cur_act_deriv;
}
                       

__kernel void matrix_sum(__global const float *mat,
                         __global float *c) {
    size_t x = get_global_id(0);
    size_t glob_size = get_global_size(0);
    size_t y = get_global_id(1);

    *c = mat[x + glob_size * y];
}

__kernel void matrix_mul(__global const float *a,
                         __global const float *b,
                         __global float *c,
                         unsigned long dim) {

    size_t x = get_global_id(0);
    size_t width = get_global_size(0);
    size_t y = get_global_id(1);

    float sum = 0.0;

    for (unsigned long i = 0; i < dim; i++) {
         sum += a[y * dim + i] * b[x + i * width];
    }
    c[x + width * y] = sum;
}

__kernel void convolve(__global const float *mat,
                       __global const float *kern,
                       __global float *res,
                       unsigned long mat_width,
                       unsigned long res_width,
                       unsigned long kern_width,
                       unsigned long kern_height) {

    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    float sum = 0.0;

    for (unsigned long r = 0; r < kern_width; r++) {
        for (unsigned long c = 0; c < kern_height; c++) {
            sum += kern[r * kern_width + c] *
                mat[y * mat_width + x + c + r * mat_width]; 
        }
    }

    res[y * res_width + x] = sum;
}
