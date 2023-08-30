open Alias
open Shape

type meta = {
    fselect : pooling;
    stride : int;
    filter_shape : mat tensor shape;
    out_shape : mat3 tensor shape;
  }

type input = mat3 tensor
type out   = mat3 tensor

type t = meta
