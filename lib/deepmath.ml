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
  mat |> Mat.map @@ ( *. ) cst

let mat_row_to_array col mat =
  Mat.to_array mat |> arr_get col

let mat_list_fold_left proc mat_list =
  List.fold_left (fun mat acc ->
      proc mat acc) mat_list []

let mat_list_flaten mlist =
  List.fold_right (fun lst flat_list_acc ->
        List.fold_right (fun num acc ->
            num :: acc) lst flat_list_acc) mlist [] 

let mat_sum mat =
  mat
  |> Mat.fold_cols (fun acc col_vec ->
         col_vec
         |> Vec.fold (fun acc num -> acc +. num) 0.
         |> (+.) acc) 0. 

(* let mat_list_flaten mat_list = *)
  

let convolve mat ~stride:stride kernel =
  (* let kern_arr = kernel |> Mat.to_array in *)
  let kern_rows = Mat.dim1 kernel in
  let kern_cols = Mat.dim2 kernel in
  let mat_rows = Mat.dim1 mat in
  let mat_cols = Mat.dim2 mat in

  let res_arr = Mat.make
                  (mat_rows - kern_rows + 1)
                  (mat_cols - kern_cols + 1) 0. |> Mat.to_array in

  let rec convolve_rec kernel stride mat r c =
    if r = mat_rows
    then ()
    else
      if c + kern_cols >= mat_cols
      then convolve_rec kernel stride mat (r + stride) 0
      else
        let dot_mat = gemm ~m:kern_rows ~n:kern_cols ~ar:r ~ac:c mat kernel in
        let sum = mat_sum dot_mat in
        res_arr.(r / stride).(c / stride) <- sum;
        convolve_rec kernel stride mat r (c + stride)
  in
  
  convolve_rec kernel stride mat 0 0;
  res_arr |> Mat.of_array

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

let mat_zero mat =
  Mat.make
    (Mat.dim1 mat)
    (Mat.dim2 mat) 0.
  
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

