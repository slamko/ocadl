open Common
open Types
open Alias
open Bigarray
open Tensor
open Ctypes
open C.Functions
open Unsigned

let sigmoid (x : float) : float =
  1. /. (1. +. exp(-. x))

let sigmoid' activation =
  activation *. (1. -. activation)

let tanh (x : float) : float =
  ((exp(2. *. x) -. 1.0)  /. (exp(2. *. x) +. 1.))

let tanh' activation =
  1. -. (activation *. activation)

let relu x =
  if x > 0.00001 then x else 0.

let relu' a =
  if a > 0.00001 then 1. else 0. 

let pooling_max a b =
  if a > b then a else b

let pooling_avarage a b =
  (a +. b) /. 2.

let arr_get index arr =
  Array.get arr index
 
let arr_print arr =
  arr |> Array.iter @@ Printf.printf "El: %f\n"

let hdtl lst = List.tl lst |> List.hd

let opt_to_bool = function
  | Some b -> b
  | None -> false

let bool_res_to_bool = function
  | Ok b -> b
  | Error err ->
     Printf.eprintf "error: %s\n" err ;
     false

let res_to_bool = function
  | Ok _ -> true
  | Error err ->
     Printf.eprintf "error: %s\n" err ;
     false

let mat_to_ba (tens : mat) =
  let input_mat = make matrix in
  setf input_mat arr (bigarray_start array2 tens.matrix) ;
  setf input_mat rows (Size_t.of_int @@ row tens.shape.dim1) ;
  setf input_mat cols (Size_t.of_int @@ col tens.shape.dim2) ;
  setf input_mat dim3 (Size_t.of_int 1) ;
  input_mat 

let vec_to_ba (tens : vec) =
  let input_mat = make matrix in
  setf input_mat arr (bigarray_start array1 tens.matrix) ;
  setf input_mat rows (Size_t.of_int 1) ;
  setf input_mat cols (Size_t.of_int @@ col tens.shape.dim1) ;
  setf input_mat dim3 (Size_t.of_int 1) ;
  input_mat 

let ba_to_vec matrix =
  bigarray_of_ptr array1 (Size_t.to_int (getf matrix cols))
    Float32 (getf matrix arr)

let ba_to_mat matrix =
  bigarray_of_ptr array2 ((Size_t.to_int (getf matrix rows)), (Size_t.to_int (getf matrix cols)))
    Float32 (getf matrix arr)

let fully_connected_ff (input : vec) (wmat : mat) (bvec : vec) meta =
  let res = make matrix in
  let inp_mat = vec_to_ba input in
  let wmat_ba = mat_to_ba wmat in
  let bmat_ba = vec_to_ba bvec in

  let actf = actf_to_enum meta.Fully_connected.activation in

  let ret  = cc_fully_conntected_ff (addr inp_mat) (addr wmat_ba) (addr bmat_ba) (addr res)
               (Signed.Long.of_int actf) in
  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Fully connected ff failed" ;
  end ;
  
  res
  |> ba_to_vec
  |> Vec.create

let fully_connected_bp (wmat : mat) (prev_act : vec)
      (act : vec) (diff : vec) (wgrad : mat) (bgrad : vec) prev_layer meta =

  let prev_diff_ba = make matrix in
  let wmat_ba = mat_to_ba wmat in
  let prev_act_ba = vec_to_ba prev_act in
  let act_ba = vec_to_ba act in
  let diff_ba = vec_to_ba diff in
  let wgrad_ba = mat_to_ba wgrad in
  let bgrad_ba = vec_to_ba bgrad in

  let actf = actf_to_enum meta.Fully_connected.activation in

  let ret  = cc_fully_conntected_bp (addr wmat_ba) (addr prev_act_ba) (addr act_ba)
               (addr diff_ba) (addr prev_diff_ba) (addr wgrad_ba) (addr bgrad_ba)
               (Signed.Long.of_int actf) prev_layer in
  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Fully connected ff failed" ;
  end ;
  
  let new_prev_diff = prev_diff_ba
                      |> ba_to_vec
                      |> Vec.create in

  let new_wgrad = wgrad_ba
                  |> ba_to_mat
                  |> Mat.wrap in

  let new_bgrad = bgrad_ba 
                  |> ba_to_vec
                  |> Vec.wrap in

  (new_prev_diff, new_wgrad, new_bgrad)

let conv2d_ff (input : mat) (kerns : mat) (bvec : vec) meta =
  let open Conv2D in
  
  let res = make matrix in
  let inp_ba = mat_to_ba input in
  let kerns_ba = mat_to_ba kerns in
  let bmat_ba = vec_to_ba bvec in

  let actf = actf_to_enum meta.act in
  let (Shape.ShapeMat out_shape) = meta.out_shape in

  let ret = cc_conv2d_ff (addr inp_ba) (addr kerns_ba) (addr bmat_ba)
              (Signed.Long.of_int actf) (ULong.of_int meta.padding) (ULong.of_int meta.stride)
              (ULong.of_int @@ col out_shape.dim2) (ULong.of_int @@ row out_shape.dim1) (addr res)
                in
  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Conv 2d ff failed" ;
  end ;
  
  res 
  |> ba_to_mat
  |> Mat.create

let conv2d_bp (kerns : mat) (prev_act : mat)
      (act : mat) (diff : mat) (kern_grad : mat) (bgrad : vec) prev_layer meta =

  let open Conv2D in

  let prev_diff_ba = make matrix in
  let kerns_ba = mat_to_ba kerns in
  let prev_act_ba = mat_to_ba prev_act in
  let act_ba = mat_to_ba act in
  let diff_ba = mat_to_ba diff in
  let kern_grad_ba = mat_to_ba kern_grad in
  let bgrad_ba = vec_to_ba bgrad in

  let actf = actf_to_enum meta.act in

  let ret  = cc_conv2d_bp (addr kerns_ba) (addr prev_act_ba) (addr act_ba)
               (addr diff_ba) (addr prev_diff_ba) (addr kern_grad_ba) (addr bgrad_ba)
               (Signed.Long.of_int actf) (ULong.of_int meta.stride) (ULong.of_int meta.padding) prev_layer in
  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Conv2D backprop failed\n" ;
  end ;
  
  let new_prev_diff = prev_diff_ba 
                      |> ba_to_mat
                      |> Mat.create in

  let new_kern_grad = kern_grad_ba 
                      |> ba_to_mat
                      |> Mat.wrap in

  let new_bgrad = bgrad_ba
                  |> ba_to_vec
                  |> Vec.wrap in

  (new_prev_diff, new_kern_grad, new_bgrad)

let pooling2d_ff (input : mat) meta =
  let open Pooling2D in
  
  let res = make matrix in
  let inp_ba = mat_to_ba input in

  let actf = pooling_to_enum meta.fselect in
  let (Shape.ShapeMat out_shape) = meta.out_shape in
  let (Shape.ShapeMat filter_shape) = meta.out_shape in

  let ret = cc_pooling2d_ff (addr inp_ba)
              (Signed.Long.of_int actf) (ULong.of_int meta.stride)
              (ULong.of_int @@ col out_shape.dim2) (ULong.of_int @@ row out_shape.dim1)
              (ULong.of_int @@ col filter_shape.dim2) (ULong.of_int @@ row filter_shape.dim1) 
              (addr res)
                in
  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Pooling 2d ff failed" ;
  end ;
  
  res 
  |> ba_to_mat
  |> Mat.create

let pooling2d_bp (prev_act : mat) (diff : mat)
      prev_layer meta =

  let open Pooling2D in

  let prev_diff_ba = make matrix in
  let prev_act_ba = mat_to_ba prev_act in
  let diff_ba = mat_to_ba diff in

  let actf = pooling_to_enum meta.fselect in
  let (Shape.ShapeMat filter_shape) = meta.out_shape in

  let ret  = cc_pooling2d_bp (addr prev_act_ba)
               (addr diff_ba) (addr prev_diff_ba)
               (Signed.Long.of_int actf) (ULong.of_int meta.stride) (ULong.of_int 0)
               (ULong.of_int @@ col filter_shape.dim2) (ULong.of_int @@ row filter_shape.dim1) 
               prev_layer in

  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Pooling2D backprop failed\n" ;
  end ;
  
  prev_diff_ba 
  |> ba_to_mat
  |> Mat.create

let mat_sub (a : mat) (b : mat) =
  let res = make matrix in
  let a_ba = mat_to_ba a in
  let b_ba = mat_to_ba b in

  let ret = cc_mat_sub (addr a_ba) (addr b_ba) (addr res) in
  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Mat sub failed" ;
  end ;
  
  getf res arr
  |> bigarray_of_ptr array2 ((Size_t.to_int (getf res rows)), (Size_t.to_int (getf res cols))) Float32
  |> Mat.create

let vec_sub (a : vec) (b : vec) =
  let res = make matrix in
  let a_ba = vec_to_ba a in
  let b_ba = vec_to_ba b in

  let ret = cc_mat_sub (addr a_ba) (addr b_ba) (addr res) in
  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Vec sub failed" ;
  end ;
  
  getf res arr
  |> bigarray_of_ptr array1 ((Size_t.to_int (getf res cols))) Float32
  |> Vec.create

let mat_scale scale (a : mat) =
  let res = make matrix in
  let a_ba = mat_to_ba a in

  let ret = cc_mat_scale (addr a_ba) (addr res) scale in
  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Mat scale failed" ;
  end ;
  
  getf res arr
  |> bigarray_of_ptr array2 ((Size_t.to_int (getf res rows)), (Size_t.to_int (getf res cols))) Float32
  |> Mat.create

let vec_scale scale (a : vec) =
  let res = make matrix in
  let a_ba = vec_to_ba a in

  let ret = cc_mat_scale (addr a_ba) (addr res) scale in
  if ret > 0
  then begin 
    Printf.eprintf "Error code: %d\n" ret ;
    failwith "Vec scale failed" ;
  end ;
  
  getf res arr
  |> bigarray_of_ptr array1 ((Size_t.to_int (getf res cols))) Float32
  |> Vec.create

let mat_flatten (mat : mat) =
  let ptr = bigarray_start array2 mat.matrix in

  bigarray_of_ptr array1 (row mat.shape.dim1 * col mat.shape.dim2) Float32 ptr
  |> Vec.wrap

let mat_flatten_bp (Row new_rows) (Col new_cols) (mat : vec) =
  let ptr = bigarray_start array1 mat.matrix in

  bigarray_of_ptr array2 (new_rows, new_cols) Float32 ptr
  |> Mat.wrap

let vec_sum (vec: vec) =
  let ba = vec_to_ba vec in
  cc_vec_sum (addr ba)
