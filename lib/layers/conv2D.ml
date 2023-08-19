open Alias
open Shape

type params = {
    kernels : mat vector;
    bias_mat : vec;
  }

type meta = {
    padding : int;
    stride : int;
    act : activation;
    deriv : deriv;
    kernel_num : int;
    out_shape : mat tensor shape;
  }

type input = mat tensor
type out = mat tensor

type t = meta * params
