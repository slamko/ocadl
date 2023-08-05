open Alias
open Shape

type meta = {
    fselect : float -> float -> float;
    fderiv : mat tensor shape -> float -> mat -> mat -> unit;
    stride : int;
    filter_shape : mat tensor shape;
    out_shape : mat vector tensor shape;
  }

type input = mat matrix tensor
type out = mat matrix tensor

type t = meta
