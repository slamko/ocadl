open Alias
open Matrix

type _ shape

val shape_size : 'a shape -> int

val make_shape_vec : Vec.shape -> vec tensor shape

val make_shape_mat : Mat.shape -> mat tensor shape

val make_shape_mat_vec : Mat.shape -> Vec.shape -> mat vector tensor shape

val make_shape_mat_mat : Mat.shape -> Mat.shape -> mat matrix tensor shape
