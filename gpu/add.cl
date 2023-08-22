__kernel void vector_sub(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    size_t i = get_global_id(0);

    c[i] = a[i] - b[i];
}

__kernel void matrix_scale(__global float *mat,
                           float scale) {
    size_t x = get_global_id(0);

    size_t x_size = get_global_size(0);
    size_t glob_size = get_global_size(0);

    size_t y_size = 0;

    size_t y = 0;
    size_t z = 0;

    if (get_work_dim() == 2) {
        y = get_global_id(1);
    } if (get_work_dim() == 3) {
        y_size = get_global_size(1);
        z = get_global_id(2);
    }

    mat[x + x_size * y + y_size * z] =
        mat[x + x_size * y + y_size * z] * scale;
}

__kernel void matrix_sub(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    size_t x = get_global_id(0);
    size_t x_size = get_global_size(0);
    size_t glob_size = get_global_size(0);

    size_t y_size = 0;

    size_t y = 0;
    size_t z = 0;

    if (get_work_dim() == 2) {
        y = get_global_id(1);
    } if (get_work_dim() == 3) {
        y_size = get_global_size(1);
        z = get_global_id(2);
    }

    c[x + x_size * y + y_size * z] =
        a[x + x_size * y + y_size * z] - b[x + x_size * y + y_size * z];
}

__kernel void matrix_add(__global const float *a,
                         __global const float *b,
                         __global float *c) {
    size_t x = get_global_id(0);
    size_t x_size = get_global_size(0);
    size_t glob_size = get_global_size(0);

    size_t y_size = 0;

    size_t y = 0;
    size_t z = 0;

    if (get_work_dim() == 2) {
        y = get_global_id(1);
    } if (get_work_dim() == 3) {
        y_size = get_global_size(1);
        z = get_global_id(2);
    }

    c[x + x_size * y + y_size * z] =
        a[x + x_size * y + y_size * z] + b[x + x_size * y + y_size * z];
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
