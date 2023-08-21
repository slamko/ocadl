open Alias
open Tensor

type _ shape = private
  | ShapeMatVec : Mat3.shape -> mat3 tensor shape
  | ShapeMat    : Mat.shape -> mat tensor shape
  | ShapeVec    : Vec.shape -> vec tensor shape

val shape_size : 'a shape -> int

val shape_eq : 'a shape -> 'a shape -> bool

val get_shape : 'a tensor -> 'a tensor shape

val make_shape_vec : Vec.shape -> vec tensor shape

val make_shape_mat : Mat.shape -> mat tensor shape

val make_shape_mat_vec : Mat3.shape -> mat3 tensor shape
