#include <CL/cl.h>
#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/bigarray.h>

#include "blac.h"
#include "deep.h"

cl_int ret;
cl_command_queue command_queue;
cl_context context;
cl_device_id device_id;
cl_program program;

CAMLprim value gemm(value a, value b) {
    CAMLparam2(a, b);

    struct caml_ba_array *amat = Caml_ba_array_val(a);
    struct caml_ba_array *bmat = Caml_ba_array_val(b);
    struct mat adata = mat_of_array(amat->data, amat->dim[0], amat->dim[1]);
    struct mat bdata = mat_of_array(bmat->data, bmat->dim[0], bmat->dim[1]);
    struct mat res_mat = mat_nil(amat->dim[0], bmat->dim[1]);

    long dims[2] = { amat->dim[0], bmat->dim[1] };

    int ret = 0;
    if ((ret = mat_gemm(context, command_queue, program, &adata, &bdata, &res_mat))) {
        printf ("Mul error %d\n", ret);
    }
    
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                             res_mat.matrix, dims));
}

CAMLprim value cc_vec_nil(value cols) {
    CAMLparam1(cols);
    long dims[1] = { Long_val(cols) };

    struct mat m = mat3_nil(1, cols, 1);
    
    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1, m.matrix, dims));
}

CAMLprim value cc_mat3_nil(value rows, value cols, value dim3) {
    CAMLparam3(rows, cols, dim3);
    long dims[3] = { Long_val(rows), Long_val(cols), Long_val(dim3) };

    struct mat m = mat3_nil(rows, cols, dim3);
    
    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 3, m.matrix, dims));
}

CAMLprim value cc_mat_nil(value rows, value cols) {
    CAMLparam2(rows, cols);
    long dims[2] = { Long_val(rows), Long_val(cols) };

    struct mat m = mat3_nil(rows, cols, 1);
    
    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2, m.matrix, dims));
}

CAMLprim value cc_vec_random(value cols) {
    CAMLparam1(cols);
    long dims[1] = { Long_val(cols) };

    struct mat m = mat3_random(1, cols, 1);
    
    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1, m.matrix, dims));
}

CAMLprim value cc_mat3_random(value rows, value cols, value dim3) {
    CAMLparam3(rows, cols, dim3);
    long dims[3] = { Long_val(rows), Long_val(cols), Long_val(dim3) };

    struct mat m = mat3_random(rows, cols, dim3);
    
    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 3, m.matrix, dims));
}

CAMLprim value cc_mat_random(value rows, value cols) {
    CAMLparam2(rows, cols);
    long dims[2] = { Long_val(rows), Long_val(cols) };

    struct mat m = mat3_random(rows, cols, 1);
    
    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2, m.matrix, dims));
}

CAMLprim value cc_fully_connected_ff(value input, value weight_mat,
                                     value bias_mat) {
    CAMLparam3(input, weight_mat, bias_mat);

    struct caml_ba_array *inmat = Caml_ba_array_val(input);
    struct caml_ba_array *wmat = Caml_ba_array_val(weight_mat);
    struct caml_ba_array *bmat = Caml_ba_array_val(bias_mat);

    struct mat indata = mat_of_array(inmat->data, 1, inmat->dim[0]);
    struct mat wdata = mat_of_array(wmat->data, wmat->dim[0], wmat->dim[1]);
    struct mat bdata = mat_of_array(bmat->data, 1, bmat->dim[0]);

    struct mat res_mat = {0};
    int ret = fully_connected_ff(context, command_queue, program,
                                 &indata, &wdata, &bdata, &res_mat);

    if (ret) {
        caml_fatal_error("Feed forward failed %d\n", ret);
    }
                                            
    long dims[1] = { wmat->dim[1] };
    
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                             res_mat.matrix, dims));
}

CAMLprim value cc_mat_add(value a, value b) {
    CAMLparam2(a, b);

    struct caml_ba_array *amat = Caml_ba_array_val(a);
    struct caml_ba_array *bmat = Caml_ba_array_val(b);

    struct mat adata = mat_of_array(amat->data, amat->dim[0], amat->dim[1]);
    struct mat bdata = mat_of_array(bmat->data, bmat->dim[0], bmat->dim[1]);
    struct mat res_mat = mat_nil(amat->dim[0], bmat->dim[1]);

    long dims[2] = { amat->dim[0], bmat->dim[1] };

    int ret = 0;
    if ((ret = mat_add(context, command_queue, program, &adata, &bdata, &res_mat))) {
        printf ("Add error %d\n", ret);
    }
    
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                             res_mat.matrix, dims));
}

CAMLprim value cc_vec_sum(value vec) {
    CAMLparam1(vec);
    int ret = 0;

    struct caml_ba_array *vec_data = Caml_ba_array_val(vec);
    struct mat vmat = mat_of_array(vec_data->data, 1, vec_data->dim[0]);

    float sum = 0.0;

    if ((ret = vec_sum(context, command_queue, program, &vmat, &sum))) {
        printf ("Vec sum error %d\n", ret);
    }
    
    CAMLreturn(caml_copy_double(sum));
}

CAMLprim value cc_vec_sub(value a, value b) {
    CAMLparam2(a, b);

    struct caml_ba_array *amat = Caml_ba_array_val(a);
    struct caml_ba_array *bmat = Caml_ba_array_val(b);

    struct mat adata = mat_of_array(amat->data, 1, amat->dim[0]);
    struct mat bdata = mat_of_array(bmat->data, 1, bmat->dim[0]);
    struct mat res_mat = mat_nil(amat->dim[0], amat->dim[0]);

    long dims[1] = { amat->dim[0] };

    int ret = 0;
    if ((ret =
         vec_sub(context, command_queue, program, &adata, &bdata, &res_mat))) {
        printf ("Vec sub error %d\n", ret);
    }
    
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                             res_mat.matrix, dims));
}

CAMLprim value cc_mat_flatten(value mat) {
    CAMLparam1(mat);
    struct caml_ba_array *mat_data = Caml_ba_array_val(mat);

    intnat new_dim[1] = { mat_data->dim[0] * mat_data->dim[1] };
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                             mat_data->data, new_dim));
}

CAMLprim value cc_gpu_init() {
    CAMLparam0();
    int ret = 0;

    ret = ocl_init(&command_queue, &context, &device_id);
    if (ret) return Val_int(ret);
    ret = load_program("add.c", &program, context, &device_id);

    return Val_int(ret);
}
