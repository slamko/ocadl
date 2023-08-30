open Alias
open Shape

type params = {
    kernels : mat;
    bias_mat : vec;
  }

type meta = {
    padding : int;
    stride : int;
    act : actf;
    out_shape : mat tensor shape;
  }

type input = mat tensor
type out = mat tensor

type t = meta * params
