open Alias
open Shape

type meta = {
    fselect : float -> float -> float;
    fderiv : mat tensor shape -> float -> mat -> mat -> unit;
    stride : int;
    filter_shape : mat tensor shape;
    out_shape : mat3 tensor shape;
  }

type input = mat3 tensor
type out   = mat3 tensor

type t = meta
