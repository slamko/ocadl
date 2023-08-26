float sigmoid(float x) {
    float val = (1.0 / (1.0 + exp(-x)));
   
    return val; 
}

float sigmoid_deriv(float act) {
    return (act * (1.0 - act));
}

float relu(float x) {
    if (x > 0.0) {
        return x;
    }

    return 0.0;
}

#define ACTF_SIGMOID 0
#define ACTF_RELU 1

void atomic_add_f(volatile __global float *addr, float val)
{
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

__kernel void dense_ff(__global __read_only const float *input,
                       __global __read_only const float *weight_mat,
                       __global __read_only const float *bias_mat,
                       __global __write_only float *res,
                       unsigned long dim,
                       unsigned long width) {

    size_t x = get_global_id(0);

    if (x >= width) {
        return;
    }

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
                       unsigned long width,
                       __global __read_write float *cache,
                       __global __read_write float *wmat_grad,
                       __global __read_write float *bmat_grad) {

    size_t x = get_global_id(0);

    if (x >= width) {
        return;
    }

    float diff = diff_mat[x];
    float cur_act = act_mat[x];
    float cur_act_deriv = sigmoid_deriv(cur_act);
     
    for (unsigned long i = 0; i < dim; i++) {
         size_t wmati = i * width + x;

         float weight = weight_mat[wmati];
         float prev_act = prev_act_mat[i];

         float dprev = diff * cur_act_deriv * weight;
         float dw = diff * cur_act_deriv * prev_act;

         wmat_grad[wmati] += dw;
         cache[wmati] = dprev;
    }

   bmat_grad[x] += diff * cur_act_deriv;
}

__kernel void conv_ff(__global __read_only const float *image,
                        __global __read_only const float *kern,
                        __global __read_only const float *bias_vec,
                        __constant unsigned long stride,
                        __constant unsigned long padding,
                        unsigned long actf, 
                        unsigned long im_width,
                        unsigned long im_height,
                        unsigned long kern_num,
                        unsigned long image_num,
                        unsigned long kern_width,
                        unsigned long kern_height,
                        unsigned long res_width,
                        unsigned long res_height,
                        __global __write_only float *res) {
    
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);

    size_t im_size = im_width * im_height;
    size_t kern_size = kern_width * kern_height;
    size_t res_size = res_width * res_height;

    if (x >= im_width || y >= im_height || z >= kern_num) {
        return;
    }

    if (x % stride || y % stride) {
        return;
    }

    float sum = 0.0;

    for (unsigned long i = 0; i < image_num; i++) {
        for (unsigned long r = 0; r < kern_width; r++) {
            for (unsigned long c = 0; c < kern_height; c++) {
                float cur_kval = kern[z * kern_size + r * kern_width + c];
                float cur_pixel = image[i * im_size + y * im_width + r * im_width + x + c];

                sum += cur_kval * cur_pixel;
            }
        }
    }

    float biased_sum = sum + bias_vec[z];

    float r = 0.0;
    switch (actf) {
    case ACTF_SIGMOID:
        r = sigmoid(biased_sum);
        break;
    case ACTF_RELU:
        r = relu(biased_sum);
        break;
    }
    
    res[z * res_size + y * res_width + x] = r; 
}
    

