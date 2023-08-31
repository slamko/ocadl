
__kernel void vector_sub(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    size_t i = get_global_id(0);

    c[i] = a[i] - b[i];
}

__kernel void matrix_scale( __global __read_only const float *mat,
                            __global __write_only float *res,
                           float scalef, unsigned long dim1, unsigned long dim2) {
    size_t x = get_global_id(0);

    if (x >= dim1) {
        return;
    }

    size_t x_size = get_global_size(0);
    size_t glob_size = get_global_size(0);

    size_t y_size = 0;

    size_t y = 0;
    size_t z = 0;

    if (get_work_dim() == 2) {
        y = get_global_id(1);
        if (y >= dim2) {
            return;
        }
    } if (get_work_dim() == 3) {
        y_size = get_global_size(1);
        y = get_global_id(1);
        z = get_global_id(2);
    }

    size_t coord = x + x_size * y + y_size * z;
    res[coord] = mat[coord] * scalef;
}

__kernel void matrix_sub(__global const float *a,
                         __global const float *b,
                         __global float *c,
                            unsigned long dim1,
                            unsigned long dim2) {
    size_t x = get_global_id(0);

    if (x >= dim1) {
        return;
    }
    
    size_t x_size = get_global_size(0);
    size_t glob_size = get_global_size(0);

    size_t y_size = 0;

    size_t y = 0;
    size_t z = 0;

    if (get_work_dim() >= 2) {
        y = get_global_id(1);
        if (y >= dim2) {
            return;
        }
 
    } if (get_work_dim() == 3) {
        y_size = get_global_size(1);
        z = get_global_id(2);
    }

    size_t coord = x + x_size * y + y_size * z;
    c[coord] = a[coord] - b[coord];
}


__kernel void matrix_add(__global const float *a,
                         __global const float *b,
                         __global float *c,
                            unsigned long dim1,
                            unsigned long dim2) {
    size_t x = get_global_id(0);

    if (x >= dim1) {
        return;
    }
    
    size_t x_size = get_global_size(0);
    size_t glob_size = get_global_size(0);

    size_t y_size = 0;

    size_t y = 0;
    size_t z = 0;

    if (get_work_dim() >= 2) {
        y = get_global_id(1);
        if (y >= dim2) {
            return;
        }
 
    } if (get_work_dim() == 3) {
        y_size = get_global_size(1);
        z = get_global_id(2);
    }

    size_t coord = x + x_size * y + y_size * z;
    c[coord] = a[coord] + b[coord];
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

__kernel void convolve(__global const float *image,
                     __global const float *kern,
                     __global float *res,
                     unsigned long kern_num,
                     unsigned long image_num,
                       unsigned long im_width,
                       unsigned long im_height,
                       unsigned long res_width,
                       unsigned long res_height,
                       unsigned long kern_width,
                       unsigned long kern_height) {

    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    float sum = 0.0;

    for (unsigned long r = 0; r < kern_width; r++) {
        for (unsigned long c = 0; c < kern_height; c++) {
            sum += kern[r * kern_width + c] *
                image[y * im_width + x + c + r * im_width]; 
        }
    }

    res[y * res_width + x] = sum;
}

__kernel void conv3(__global const float *image,
                     __global const float *kern,
                     __global float *res,
                     unsigned long kern_num,
                     unsigned long image_num,
                       unsigned long im_width,
                       unsigned long im_height,
                       unsigned long res_width,
                       unsigned long res_height,
                       unsigned long kern_width,
                       unsigned long kern_height) {
 
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);

    size_t im_size = im_width * im_height;
    size_t kern_size = kern_width * kern_height;
    size_t res_size = res_width * res_height;

    size_t locx = get_local_id(0);
    size_t locy = get_local_id(1);

    __local float loc_img[32 * 32];

    /*
    for (size_t r = 0; r < kern_height; r++) {
        for (size_t c = 0; c < kern_width; c++) {

            if ((x + c < im_width && y + r < im_height && z < kern_num) &&
                ((locx == 0 && locy == 0) ||
                 (locx > 0 && locy == 0 && c == kern_width - 1) ||
                 (locx == 0 && locy > 0 && r == kern_height - 1) ||
                 (locx > 0 && locy > 0 && c == kern_width - 1 && r == kern_height - 1))) {   

                float cur_pixel = image[y * im_width + r * im_width + x + c];
                printf("Sample: x = %lu, y = %lu, r = %lu, c = %lu, %f\n", locx, locy, r, c, cur_pixel);
                
                loc_img[locy * 32 + r * 32 + locx + c] = cur_pixel;
            }
        }
    }

    if (locx == 0 && locy == 0) {
        for (size_t r = 0; r < 32; r++) {
            for (size_t c = 0; c < 32; c++) {

                if (r >= im_height || c >= im_width) {
                    break;
                }   

               float cur_pixel = image[r * im_width + c];
               loc_img[r * 32 + c] = cur_pixel;
            }
        }
    }
    */

    loc_img[locy * 32 + locx] = image[y * im_width + x];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x >= im_width || y >= im_height || z >= kern_num) {
        return;
    }

    float sum = 0.0;

    for (unsigned long i = 0; i < image_num; i++) {
        for (unsigned long r = 0; r < kern_width; r++) {
            for (unsigned long c = 0; c < kern_height; c++) {
                float cur_kval = kern[z * kern_size + r * kern_width + c];
                float cur_pixel = loc_img[i * im_size + locy * 32 + r * 32 + locx + c];

                sum += cur_kval * cur_pixel;
            }
        }
    }
    
    res[z * res_size + y * res_width + x] = sum;
}

#define WG_SIZE 256

__kernel void conv2(__global const float *image,
                     __global const float *kern,
                     __global float *res,
                     __local float *cache,
                     unsigned long kern_num,
                     unsigned long image_num,
                     unsigned long im_width,
                     unsigned long im_height,
                     unsigned long res_width,
                     unsigned long res_height,
                     unsigned long cache_width,
                     unsigned long cache_height,
                     unsigned long kern_width,
                     unsigned long kern_height) {
 
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t z = get_global_id(2);

    size_t im_size = im_width * im_height;
    size_t kern_size = kern_width * kern_height;
    size_t res_size = res_width * res_height;

    size_t locx = get_local_id(0);
    size_t locy = get_local_id(1);

    size_t cache_size = cache_height * cache_width;
    size_t epoch = cache_size / WG_SIZE;

    if (cache_size % WG_SIZE) {
        epoch += 1;
    }
   
    for (size_t i = 0; i < epoch; i++) {
        size_t id = i * (get_local_size(0) * get_local_size(1)) + (locy * cache_width + locx);

        if (x >= im_width || y >= im_height || z >= kern_num) {
            continue;
        }
    
        if (id < cache_size) {
            float cur_pixel = image[((i * WG_SIZE) / cache_width) * im_width + y * im_width + x + ((i * WG_SIZE) % cache_width)];
            cache[id] = cur_pixel;
        } 
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x >= im_width || y >= im_height || z >= kern_num) {
        return;
    }

    float sum = 0.0;

        for (unsigned long r = 0; r < kern_width; r++) {
            for (unsigned long c = 0; c < kern_height; c++) {
                float cur_kval = kern[r * kern_width + c];
                float cur_pixel = cache[locy * cache_width + r * cache_width + locx + c];

                sum += cur_kval * cur_pixel;
            }
        }
    
    res[y * res_width + x] = sum;
}
