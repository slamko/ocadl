open Alias
open Shape

type params = {
    kernels : mat3;
    bias_mat : vec;
  }

type meta = {
    padding : int;
    stride : int;
    act : activation;
    deriv : deriv;
    kernel_num : int;
    out_shape : mat3 tensor shape;
  }

type input = mat3 tensor
type out = mat3 tensor

type t = meta * params
