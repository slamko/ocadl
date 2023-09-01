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

float relu_deriv(float x) {
    if (x > 0.0) return 1.0;

    return 0.0;
}

#define ACTF_SIGMOID 0
#define ACTF_RELU 1


float deriv(float value, long actf) {
    switch (actf) {
    case ACTF_SIGMOID:
        return sigmoid_deriv(value);
        break;
    case ACTF_RELU:
        return relu_deriv(value);
    default:
        printf("Error: Unknown activation function\n");
        break;
    }
}

#define POOLING_MAX 0
#define POOLING_AVG 1

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
                       unsigned long width,
                       long actf) {

    size_t x = get_global_id(0);

    if (x >= width) {
        return;
    }

    float sum = 0.0;

    for (unsigned long i = 0; i < dim; i++) {
         sum += input[i] * weight_mat[x + i * width];
    }

    float biased_sum = sum + bias_mat[x];
    float r = 0.0;

    switch (actf) {
    case ACTF_SIGMOID:
        r = sigmoid(biased_sum);
        break;
    case ACTF_RELU:
        r = relu(biased_sum);
        break;
    default:
        printf("Error: Unknown activation function\n");
        break;
    }
    
    res[x] = r;
}

__kernel void dense_bp(__global __read_only const float *weight_mat,
                       __global __read_only const float *prev_act_mat,
                       __global __read_only const float *act_mat,
                       __global __read_only const float *diff_mat,
                       unsigned long dim,
                       unsigned long width,
                       __global __read_write float *cache,
                       __global float *wmat_grad,
                       __global float *bmat_grad,
                       long actf) {

    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    if (x >= width || y >= dim) {
        return;
    }

    float diff = diff_mat[x];
    float cur_act = act_mat[x];

    float cur_act_deriv = deriv(cur_act, actf);
    
    size_t wmati = y * width + x;

    float weight = weight_mat[wmati];
    float prev_act = prev_act_mat[y];

    float dprev = diff * cur_act_deriv * weight;
    float dw = diff * cur_act_deriv * prev_act;

    wmat_grad[wmati] += dw;
    cache[wmati] = dprev;

    bmat_grad[x] += diff * cur_act_deriv;
}

__kernel void mat_sum(__global __read_only float *mat,
                        unsigned long rows,
                        unsigned long cols,
                        __global __write_only float *res) {

    size_t x = get_global_id(0);

    if (x >= rows) {
        return;
    }

    float sum = 0.0;
    for (unsigned int i = 0; i < cols; i++) {
        sum += mat[x * cols + i];  
    }

    res[x] = sum;
}

__kernel void conv_ff(__global __read_only const float *image,
                        __global __read_only const float *kern,
                        __global __read_only const float *bias_vec,
                        unsigned long stride,
                        unsigned long padding,
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
    default:
        printf("Error: Unknown activation function\n");
        break;
    }
    
    res[z * res_size + y * res_width + x] = r; 
}

__kernel void conv_deriv_bp(__global __read_only const float *act,
                        __global __read_only const float *diff_mat,
                        unsigned long actf, 
                        unsigned long act_width,
                        unsigned long act_height,
                        __global __write_only float *dz) {

    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);

    size_t act_size = act_width * act_height;

    size_t id = z * act_size + y * act_width + x;
    float dact = deriv(act[id], actf);
    dz[id] = dact * diff_mat[id];
}
    
__kernel void conv_bp(__global __read_only const float *act_prev,
                        __global __read_only const float *dz,
                        unsigned long stride,
                        unsigned long padding,
                        unsigned long actf, 
                        unsigned long act_width,
                        unsigned long act_height,
                        unsigned long act_num,
                        unsigned long prev_act_width,
                        unsigned long prev_act_height,
                        unsigned long prev_act_num,
                        unsigned long kern_width,
                        unsigned long kern_height,
                        __global __read_write float *kern_grad) {
    
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);

    size_t act_size = act_width * act_height;
    size_t prev_act_size = prev_act_width * prev_act_height;
    size_t kern_size = kern_width * kern_height;

    if (x >= kern_width || y >= kern_height) {
        return;
    }

    float sum = 0.0;
    for (unsigned long i = 0; i < prev_act_num; i++) {
        for (unsigned long r = 0; r < act_width; r++) {
            for (unsigned long c = 0; c < act_height; c++) {
                float cur_kval = dz[z * act_size + r * act_width + c];
                float cur_pixel = act_prev[i * prev_act_size + y * prev_act_width + r * prev_act_width + x + c];

                sum += cur_kval * cur_pixel;
            }
        }
    }

    kern_grad[z * kern_size + y * kern_width + x] += sum; 
}

__kernel void pooling_ff(__global __read_only const float *image,
                         unsigned long stride,
                         unsigned long filter_width,
                         unsigned long filter_height,
                         unsigned long im_width,
                         unsigned long im_height,
                         unsigned long im_num,
                         unsigned long pooling_type,
                         unsigned long res_width,
                         unsigned long res_height,
                        __global __write_only float *res) {
    
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);

    size_t im_size = im_width * im_height;
    size_t filter_size = filter_width * filter_height;
    size_t res_size = res_width * res_height;

    float res_val = 0.0;

    switch (pooling_type) {
    case POOLING_MAX:
        for (unsigned long r = 0; r < filter_width; r++) {
            for (unsigned long c = 0; c < filter_height; c++) {
                float cur_pixel = image[z * im_size + y * im_width + r * im_width + x + c];

                if (cur_pixel > res_val) {
                    res_val = cur_pixel;
                }
            }
        }

        break;
    case POOLING_AVG:
        for (unsigned long r = 0; r < filter_width; r++) {
            for (unsigned long c = 0; c < filter_height; c++) {
                float cur_pixel = image[z * im_size + y * im_width + r * im_width + x + c];
                res_val += cur_pixel;
            }
        }

        res_val /= filter_size;
        break;
    default:
        printf("Error: Unknown pooling type\n");
        break;
    }

    res[z * res_size + y * res_width + x] = res_val; 
}

__kernel void pooling_bp(__global __read_only const float *act_prev,
                        __global __read_only const float *diff_mat,
                        unsigned long stride,
                        unsigned long padding,
                        unsigned long type, 
                        unsigned long diff_width,
                        unsigned long diff_height,
                        unsigned long filter_width,
                        unsigned long filter_height,
                        unsigned long prev_act_width,
                        unsigned long prev_act_height,
                        unsigned long prev_act_num,
                        __global __write_only float *prev_diff) {
    
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);

    size_t diff_size = diff_width * diff_height;
    size_t prev_act_size = prev_act_width * prev_act_height;

    if (x >= diff_width || y >= diff_height) {
        return;
    }

    switch (type) {
    case POOLING_MAX:;
        float pix_max = -1000.0; 
        size_t max_id = 0;

        for (unsigned long r = 0; r < filter_width; r++) {
            for (unsigned long c = 0; c < filter_height; c++) {
                size_t id = z * prev_act_size + y * prev_act_width + r * prev_act_width + x + c;
                float cur_pixel = act_prev[id];

                if (cur_pixel > pix_max) {
                    pix_max = cur_pixel;
                    max_id = id; 
                }
            }
        }

        prev_diff[max_id] = diff_mat[z * diff_size + y * diff_width + x];

        break;
    case POOLING_AVG:
    /*
        for (unsigned long r = 0; r < filter_width; r++) {
            for (unsigned long c = 0; c < filter_height; c++) {
                float cur_pixel = image[z * im_size + y * im_width + r * im_width + x + c];
                res_val += cur_pixel;
            }
        }

        res_val /= filter_size;
    */
        break;
    default:
        printf("Error: Unknown pooling type\n");
        break;
    }
}


