open Alias
open Shape

type params = {
    weight_mat : mat;
    bias_mat : vec;
  }

type meta = {
    activation : activation;
    derivative : deriv;
    out_shape : vec tensor shape;
  }

type input  = vec tensor
type out    = vec tensor

type t = meta * params
