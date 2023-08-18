open Alias
open Shape

type params = {
    weight_mat : mat;
    bias_mat : mat;
  }
[@@deriving show]

type meta = {
    activation : activation;
    derivative : deriv;
    out_shape : vec tensor shape;
  }

type input = float vector tensor
type out = float vector tensor

type t = meta * params
