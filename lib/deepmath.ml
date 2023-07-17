open Lacaml.D
(* open Types *)

let sigmoid (x : float) : float =
  1. /. (1. +. exp(-. x))


let make_zero_mat_list mat_list =
  List.fold_right (fun mat mlist ->
      (Mat.make (Mat.dim1 mat) (Mat.dim2 mat) 0.) ::  mlist) mat_list []


let arr_get index arr =
  Array.get arr index

let mat_add mat1 mat2 =
  Mat.add mat1 mat2

let mat_sub mat1 mat2 =
  Mat.sub mat1 mat2

let mat_add_const cst mat =
  Mat.add_const cst mat

let mat_scale cst mat =
  Mat.map (fun v -> v *. cst) mat

let mat_row_to_array col mat =
  Mat.to_array mat |> arr_get col

let mat_list_fold_left proc mat_list =
  List.fold_left (fun mat acc ->
      proc mat acc) mat_list []

let mat_print (mat : mat)  =
   Format.printf
    "\
      @[<2>Matrix :\n\
        @\n\
        %a@]\n\
      @\n"
    Lacaml.Io.pp_fmat mat

let arr_print arr =
  arr |> Array.iter @@ Printf.printf "El: %f\n"

