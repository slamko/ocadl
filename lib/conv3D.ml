open Alias
open Shape

type params = {
    kernels : mat matrix;
    bias_mat : mat;
  }

type meta = {
    padding : int;
    stride : int;
    act : activation;
    deriv : deriv;
    kernel_num : int;
    out_shape : mat vector tensor shape;
  }

type input = mat matrix tensor
type out = mat matrix tensor

type t = meta * params
