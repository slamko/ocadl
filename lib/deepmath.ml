open Lacaml.D
(* open Types *)

let sigmoid (x : float) : float =
  1. /. (1. +. exp(-. x))

let sigmoid' activation =
  activation *. (1. -. activation)

let tanh (x : float) : float =
  ((exp(2. *. x) -. 1.0)  /. (exp(2. *. x) +. 1.))

let tanh' activation =
  1. -. (activation *. activation)

let relu x =
  if x > 0. then x else 0.

let relu' a =
  if a > 0. then 1. else 0. 

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

let mat_list_flaten mlist =
  List.fold_right (fun lst flat_list_acc ->
        List.fold_right (fun num acc ->
            num :: acc) lst flat_list_acc) mlist [] 

let mat_flaten mat =
  []
  |> List.cons
       (mat
        |> Mat.to_list
        |> List.flatten)
  |> Mat.of_list

let mat_reshape mat nrows ncols =
  let mlist = Mat.to_list mat in
  let flat_mlist = List.flatten mlist in

  let rec reshape_rec flat_mlist i reshape_acc cur_acc =
    match flat_mlist with
    | [] -> reshape_acc
    | h::t ->
       let cur_col = h :: cur_acc in
       if (i mod ncols) = 0
       then reshape_rec t (i + 1) (cur_col::reshape_acc) []
       else reshape_rec t (i + 1) reshape_acc cur_col
  in

  reshape_rec flat_mlist 1 [] []
  |> Mat.of_list
  
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

