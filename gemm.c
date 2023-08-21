#include <caml/mlvalues.h>
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/bigarray.h>

void hello_mat(void);

CAMLprim value gemm(value a, value b) {
    CAMLparam2(a, b);
    struct caml_ba_array *amat = Caml_ba_array_val(a);
    float *adata = amat->data;

    for (size_t i = 0; i < amat->dim[0]; i++) {
        adata[i] += 2.0;
    }

    CAMLreturn(Val_unit);
}


CAMLprim value mat_make(value rows, value cols) {
    CAMLparam2(rows, cols);
    long dims[2] = { Long_val(cols), Long_val(rows) };

    float *arr = calloc(rows * cols, sizeof *arr);
    hello_mat();

    CAMLreturn(
        caml_ba_alloc(CAML_BA_C_LAYOUT | CAML_BA_FLOAT32, 2, arr, dims));
}
