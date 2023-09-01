#include <CL/cl.h>
#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/fail.h>
#include <caml/bigarray.h>

#include "ocl.h"
#include "blasc.h"
#include "deep.h"

struct mat mat_of_ba(struct caml_ba_array *ba) {
    struct mat mat = {0};

    if (ba->num_dims == 1) {
        mat = mat_of_array(ba->data, 1, ba->dim[0]);
    } else if (ba->num_dims == 2) {
        mat = mat_of_array(ba->data, ba->dim[0], ba->dim[1]);
    } else if (ba->num_dims == 3) {
        mat = mat3_of_array(ba->data, ba->dim[0], ba->dim[1], ba->dim[2]);
    }

    return mat;
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

CAMLprim value cc_fully_connected_ff(value input, value weight_mat,
                                     value bias_mat, value actf_val) {
    CAMLparam3(input, weight_mat, bias_mat);

    long actf = Long_val(actf_val);

    struct caml_ba_array *inmat = Caml_ba_array_val(input);
    struct caml_ba_array *wmat = Caml_ba_array_val(weight_mat);
    struct caml_ba_array *bmat = Caml_ba_array_val(bias_mat);

    struct mat indata = mat_of_ba(inmat);
    struct mat wdata = mat_of_ba(wmat);
    struct mat bdata = mat_of_ba(bmat);

    struct mat res_mat = {0};
    int ret = fully_connected_ff(&indata, &wdata, &bdata, &res_mat, actf);

    if (ret) {
        caml_fatal_error("Feed forward failed %d\n", ret);
    }
                                            
    long dims[1] = { res_mat.cols };

    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                             res_mat.matrix, dims));
}

CAMLprim value cc_conv2d_bp_native(value weight_mat, value prev_act_mat,
                                            value act_mat, value diff_mat,
                                            value wgrad_mat, value bgrad_mat,
                                            value prev_layer_val, value actf_val) {

    CAMLparam5(weight_mat, prev_act_mat, act_mat, diff_mat, wgrad_mat);
    CAMLxparam3(bgrad_mat, prev_layer_val, actf_val);
    CAMLlocal1(res_tuple);

    struct caml_ba_array *dmat = Caml_ba_array_val(diff_mat);
    struct caml_ba_array *wmat = Caml_ba_array_val(weight_mat);
    _Bool prev_layer = Bool_val(prev_layer_val);
    long actf = Long_val(actf_val);

    struct caml_ba_array *wgrad_ba = Caml_ba_array_val(wgrad_mat);
    struct caml_ba_array *bgrad_ba = Caml_ba_array_val(bgrad_mat);

    struct caml_ba_array *actmat = Caml_ba_array_val(act_mat);
    struct caml_ba_array *prev_actmat = Caml_ba_array_val(prev_act_mat);

    struct mat diff_data = mat_of_ba(dmat);
    struct mat wdata = mat_of_ba(wmat);
    struct mat act_data = mat_of_ba(actmat);
    struct mat prev_act_data = mat_of_ba(prev_actmat);

    struct mat wgrad = mat_of_ba(wgrad_ba);
    struct mat bgrad = mat_of_ba(bgrad_ba);

    struct mat prev_diff;

    int ret = conv_bp(&wdata, &prev_act_data, &act_data,
                      &diff_data, &prev_diff, &wgrad, &bgrad, actf, 1, 0, prev_layer);

    if (ret) {
        caml_fatal_error("Conv back prop failed %d\n", ret);
    }
                                            
    res_tuple = caml_alloc_tuple(3);

    long pd_dims[1] = { prev_diff.cols };
    
    Store_field(res_tuple, 0,
                caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                              prev_diff.matrix, pd_dims));

    long wg_dims[2] = { wgrad.rows, wgrad.cols };

    Store_field(res_tuple, 1,
                caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                              wgrad.matrix, wg_dims));
 
    long bg_dims[1] = { bgrad.cols };

    Store_field(res_tuple, 2,
                caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                              bgrad.matrix, bg_dims));
    CAMLreturn(res_tuple);
}

CAMLprim value cc_conv_bp_bytecode(value *argv, int argn) {
    if (argn != 8) {
        caml_fatal_error("Wrong number of args for C stub");
    }
    
    return cc_conv2d_bp_native(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
}

CAMLprim value cc_fully_connected_bp_native(value weight_mat, value prev_act_mat,
                                            value act_mat, value diff_mat,
                                            value wgrad_mat, value bgrad_mat,
                                            value prev_layer_val, value actf_val) {

    CAMLparam5(weight_mat, prev_act_mat, act_mat, diff_mat, wgrad_mat);
    CAMLxparam3(bgrad_mat, prev_layer_val, actf_val);
    CAMLlocal1(res_tuple);

    struct caml_ba_array *dmat = Caml_ba_array_val(diff_mat);
    struct caml_ba_array *wmat = Caml_ba_array_val(weight_mat);
    _Bool prev_layer = Bool_val(prev_layer_val);
    long actf = Long_val(actf_val);

    struct caml_ba_array *wgrad_ba = Caml_ba_array_val(wgrad_mat);
    struct caml_ba_array *bgrad_ba = Caml_ba_array_val(bgrad_mat);

    struct caml_ba_array *actmat = Caml_ba_array_val(act_mat);
    struct caml_ba_array *prev_actmat = Caml_ba_array_val(prev_act_mat);

    struct mat diff_data = mat_of_ba(dmat);
    struct mat wdata = mat_of_ba(wmat);
    struct mat act_data = mat_of_ba(actmat);
    struct mat prev_act_data = mat_of_ba(prev_actmat);

    struct mat wgrad = mat_of_ba(wgrad_ba);
    struct mat bgrad = mat_of_ba(bgrad_ba);

    struct mat prev_diff;

    int ret = fully_connected_bp(&wdata, &prev_act_data, &act_data,
                                 &diff_data, &prev_diff, &wgrad, &bgrad, actf, prev_layer);

    if (ret) {
        caml_fatal_error("Fully connected backpropagation failed %d\n", ret);
    }
                                            
    res_tuple = caml_alloc_tuple(3);

    long pd_dims[1] = { prev_diff.cols };
    
    Store_field(res_tuple, 0,
                caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                              prev_diff.matrix, pd_dims));

    long wg_dims[2] = { wgrad.rows, wgrad.cols };

    Store_field(res_tuple, 1,
                caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                              wgrad.matrix, wg_dims));
 
    long bg_dims[1] = { bgrad.cols };

    Store_field(res_tuple, 2,
                caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                              bgrad.matrix, bg_dims));
    CAMLreturn(res_tuple);
}

CAMLprim value cc_fully_connected_bp_bytecode(value *argv, int argn) {
    if (argn != 8) {
        caml_fatal_error("Wrong number of args for C stub");
    }
    
    return cc_fully_connected_bp_native(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
}

CAMLprim value cc_conv_ff_native(value input, value kernels,
                                 value bias_vec, value actf_val,
                                 value padding_val, value stride_val,
                                 value res_width_val, value res_height_val) {
    CAMLparam5(input, kernels, bias_vec, actf_val, padding_val);
    CAMLxparam3(stride_val, res_width_val, res_height_val);

    struct caml_ba_array *inp_arr = Caml_ba_array_val(input);
    struct caml_ba_array *kernels_arr = Caml_ba_array_val(kernels);
    struct caml_ba_array *bvec_arr = Caml_ba_array_val(bias_vec);

    long actf = Long_val(actf_val);
    long padding = Long_val(padding_val);
    long stride = Long_val(stride_val);
    long res_width = Long_val(res_width_val);
    long res_height = Long_val(res_height_val);

    struct mat inp_mat = mat_of_ba(inp_arr);
    struct mat kernels_mat = mat_of_ba(kernels_arr);
    struct mat bvec = mat_of_ba(bvec_arr);
    
    struct mat res_mat = {0};

    int ret = conv_ff(&inp_mat, &kernels_mat, &bvec,
                      actf, padding, stride, res_width, res_height, &res_mat);

    if (ret) {
        caml_fatal_error("Conv feed forward failed %d\n", ret);
    }
                                            
    long dims[3] = { res_width, res_height, kernels_mat.dim3 };

    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 3,
                      res_mat.matrix, dims));
}

CAMLprim value cc_conv_ff_bytecode(value *argv, int argn) {
    if (argn != 8) {
        caml_fatal_error("Wrong number of args for C stub");
    }
    
    return cc_conv_ff_native(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
}

CAMLprim value cc_pooling_ff_native(value input,
                                    value type_val, value stride_val,
                                    value res_width_val, value res_height_val,
                                    value filterw_val, value filterh_val) {

    CAMLparam5(input, type_val, stride_val, res_width_val, res_height_val);
    CAMLxparam2(filterw_val, filterh_val);

    struct caml_ba_array *inp_arr = Caml_ba_array_val(input);

    long stride = Long_val(stride_val);
    long res_width = Long_val(res_width_val);
    long res_height = Long_val(res_height_val);

    long type = Long_val(type_val);
    long filter_width = Long_val(filterw_val);
    long filter_height = Long_val(filterh_val);

    struct mat inp_mat = mat_of_ba(inp_arr);
    
    struct mat res_mat = {0};

    int ret = pooling_ff(&inp_mat, type, stride, res_width, res_height,
                         filter_width, filter_height, &res_mat);

    if (ret) {
        caml_fatal_error("Pooling feed forward failed %d\n", ret);
    }
                                            
    long dims[2] = { res_width, res_height };

    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                      res_mat.matrix, dims));
}

CAMLprim value cc_pooling_ff_bytecode(value *argv, int argn) {
    if (argn != 7) {
        caml_fatal_error("Wrong number of args for C stub");
    }
    
    return cc_pooling_ff_native(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
}

CAMLprim value cc_pooling2d_bp_native(value prev_act_val, value diff_mat_val,
                                    value type_val, value stride_val,
                                    value filterw_val, value filterh_val) {

    CAMLparam5(prev_act_val, diff_mat_val, type_val, stride_val, filterw_val);
    CAMLxparam1(filterh_val);

    struct caml_ba_array *prev_act_arr = Caml_ba_array_val(prev_act_val);
    struct caml_ba_array *diff_arr = Caml_ba_array_val(diff_mat_val);

    long stride = Long_val(stride_val);

    long type = Long_val(type_val);
    long filter_width = Long_val(filterw_val);
    long filter_height = Long_val(filterh_val);

    struct mat prev_act_mat = mat_of_ba(prev_act_arr);
    struct mat diff_mat = mat_of_ba(diff_arr);
    
    struct mat prev_diff_mat = {0};

    int ret = pooling_bp(&prev_act_mat, &diff_mat, &prev_diff_mat, type, stride, 0,
                         filter_width, filter_height, true);

    if (ret) {
        caml_fatal_error("Pooling feed forward failed %d\n", ret);
    }
                                            
    long dims[2] = { prev_diff_mat.rows, prev_diff_mat.cols };

    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                      prev_diff_mat.matrix, dims));
}

CAMLprim value cc_pooling_bp_bytecode(value *argv, int argn) {
    if (argn != 2) {
        caml_fatal_error("Wrong number of args for C stub");
    }
    
    return cc_pooling_ff_native(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
}

CAMLprim value cc_vec_scale(double scale, value mat) {
    /* CAMLparam1(mat); */

    struct caml_ba_array *mat_arr = Caml_ba_array_val(mat);

    struct mat mat_mat = mat_of_array(mat_arr->data, 1, mat_arr->dim[0]);
    struct mat res_mat = mat_nil(1, mat_arr->dim[0]);

    long dims[1] = { mat_arr->dim[0] };

    int ret = 0;
    if ((ret = mat_scale(&mat_mat, &res_mat, scale))) {
        error ("Error code: %d\n", ret);
        caml_failwith("Vec scale error\n");
    }
    
    return caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                             res_mat.matrix, dims);
}

CAMLprim value cc_vec_scale_byte(value scale, value mat) {
    caml_fatal_error("No bytecode support\n");
}

CAMLprim value cc_mat_scale(value scale, value mat) {
    CAMLparam2(scale, mat);

    struct caml_ba_array *mat_arr = Caml_ba_array_val(mat);

    struct mat mat_mat = mat_of_ba(mat_arr);

    long dims[2] = { mat_arr->dim[0], mat_arr->dim[1] };
    struct mat res_mat = {0};

    int ret = 0;
    if ((ret = mat_scale(&mat_mat, &res_mat, Double_val(scale)))) {
        error ("Error code: %d\n", ret);
        caml_failwith("Mat scale error\n");
    }
    
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                             res_mat.matrix, dims));
}

CAMLprim value cc_mat_add(value a, value b) {
    CAMLparam2(a, b);

    struct caml_ba_array *amat = Caml_ba_array_val(a);
    struct caml_ba_array *bmat = Caml_ba_array_val(b);

    struct mat adata = mat_of_ba(amat);
    struct mat bdata = mat_of_ba(bmat);
    struct mat res_mat = {0};

    long dims[2] = { amat->dim[0], bmat->dim[1] };

    int ret = 0;
    if ((ret = mat_add(&adata, &bdata, &res_mat))) {
        error ("Error code: %d\n", ret);
        caml_failwith("Mat add error\n");
    }
    
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                             res_mat.matrix, dims));
}

CAMLprim value cc_vec_sum(value vec) {
    CAMLparam1(vec);
    int ret = 0;

    struct caml_ba_array *vec_data = Caml_ba_array_val(vec);
    struct mat vmat = mat_of_ba(vec_data);

    float sum = 0.0;

    if ((ret = vec_sum(&vmat, &sum))) {
        printf ("Vec sum error %d\n", ret);
    }
    
    CAMLreturn(caml_copy_double(sum));
}

CAMLprim value cc_vec_add(value a, value b) {
    CAMLparam2(a, b);

    struct caml_ba_array *amat = Caml_ba_array_val(a);
    struct caml_ba_array *bmat = Caml_ba_array_val(b);

    struct mat adata = mat_of_ba(amat);
    struct mat bdata = mat_of_ba(bmat);
    struct mat res_mat = {0};

    long dims[1] = { amat->dim[0] };

    int ret = 0;
    if ((ret =
         mat_add(&adata, &bdata, &res_mat))) {
        printf ("Vec sub error %d\n", ret);
    }
    
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                             res_mat.matrix, dims));
}

CAMLprim value cc_mat_sub(value a, value b) {
    CAMLparam2(a, b);

    struct caml_ba_array *amat = Caml_ba_array_val(a);
    struct caml_ba_array *bmat = Caml_ba_array_val(b);

    struct mat adata = mat_of_ba(amat);
    struct mat bdata = mat_of_ba(bmat);
    struct mat res_mat;

    long dims[2] = { amat->dim[0], amat->dim[1] };

    int ret = 0;
    if ((ret =
         mat_sub(&adata, &bdata, &res_mat))) {

        error("Error code: %d\n", ret);
        caml_failwith ("Mat sub error\n");
    }
    
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                             res_mat.matrix, dims));
}

CAMLprim value cc_vec_sub(value a, value b) {
    CAMLparam2(a, b);

    struct caml_ba_array *amat = Caml_ba_array_val(a);
    struct caml_ba_array *bmat = Caml_ba_array_val(b);

    struct mat adata = mat_of_ba(amat);
    struct mat bdata = mat_of_ba(bmat);
    struct mat res_mat;

    long dims[1] = { amat->dim[0] };

    int ret = 0;
    if ((ret =
         mat_sub(&adata, &bdata, &res_mat))) {

        error("Error code: %d\n", ret);
        caml_failwith("Vec sub error\n");
    }

    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                             res_mat.matrix, dims));
}

CAMLprim value cc_mat3_flatten(value mat) {
    CAMLparam1(mat);
    struct caml_ba_array *mat_data = Caml_ba_array_val(mat);

    intnat new_dim[1] = { mat_data->dim[0] * mat_data->dim[1] * mat_data->dim[2]};
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                             mat_data->data, new_dim));
}

CAMLprim value cc_mat3_flatten_bp(value rows, value cols, value dim3, value mat) {
    CAMLparam3(rows, cols, mat);
    struct caml_ba_array *mat_data = Caml_ba_array_val(mat);

    intnat new_dim[3] = { Long_val(rows), Long_val(cols), Long_val(dim3) };
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 3,
                             mat_data->data, new_dim));
}

CAMLprim value cc_mat_flatten(value mat) {
    CAMLparam1(mat);
    struct caml_ba_array *mat_data = Caml_ba_array_val(mat);

    intnat new_dim[1] = { mat_data->dim[0] * mat_data->dim[1] };
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 1,
                             mat_data->data, new_dim));
}

CAMLprim value cc_mat_flatten_bp(value rows, value cols, value mat) {
    CAMLparam3(rows, cols, mat);
    struct caml_ba_array *mat_data = Caml_ba_array_val(mat);

    intnat new_dim[2] = { Long_val(rows), Long_val(cols) };
    CAMLreturn(caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2,
                             mat_data->data, new_dim));
}

CAMLprim value cc_mat_free(value mat_arr) {
    CAMLparam1(mat_arr);
    struct caml_ba_array *mat_data = Caml_ba_array_val(mat_arr);
    struct mat mat = mat_of_ba(mat_data);

    mat_free(&mat);

    CAMLreturn(Val_unit);
}

CAMLprim value cc_mat_print(value mat_arr) {
    CAMLparam1(mat_arr);
    struct caml_ba_array *mat_data = Caml_ba_array_val(mat_arr);
    struct mat mat = mat_of_ba(mat_data);

    mat_print(&mat);

    CAMLreturn(Val_unit);
}

CAMLprim value cc_gpu_finish() {
    CAMLparam0();
    int ret = 0;
    
    ret = ocl_finish();

    CAMLreturn(Val_unit);
}

CAMLprim value cc_gpu_init() {
    CAMLparam0();

    int ret = ocl_init();
    if (ret) {
        error("Error code: %d\n", ret);
        caml_failwith("OpenCL initialization failed\n");
    }

    CAMLreturn(Val_unit);
}
