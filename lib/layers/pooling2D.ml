open Alias
open Shape

type meta = {
    fselect : pooling;
    stride : int;
    filter_shape : mat tensor shape;
    out_shape : mat tensor shape;
  }

type input = mat tensor
type out = mat tensor

type t = meta
