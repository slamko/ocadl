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
  
  getf res arr
  |> bigarray_of_ptr array1 (Size_t.to_int (getf res cols)) Float32
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
  
  let new_prev_diff = getf prev_diff_ba arr
                      |> bigarray_of_ptr array1 (Size_t.to_int (getf prev_diff_ba cols)) Float32
                      |> Vec.create in

  let new_wgrad = getf wgrad_ba arr
                   |> bigarray_of_ptr array2 ((Size_t.to_int (getf wgrad_ba rows)), (Size_t.to_int (getf wgrad_ba cols))) Float32
                   |> Mat.wrap in

  let new_bgrad = getf bgrad_ba arr
                   |> bigarray_of_ptr array1 ((Size_t.to_int (getf bgrad_ba cols))) Float32
                   |> Vec.wrap in

  (new_prev_diff, new_wgrad, new_bgrad)


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

(*

external cc_mat3_nil : int -> int -> int -> Mat3.tensor = "cc_mat3_nil"

external cc_mat_nil : int -> int -> Mat.tensor = "cc_mat_nil"

external cc_vec_nil : int -> Vec.tensor = "cc_vec_nil"

external cc_fully_connected_ff : Vec.tensor -> Mat.tensor -> Vec.tensor -> int ->
  Vec.tensor = "cc_fully_connected_ff"

external cc_mat_add : Mat.tensor -> Mat.tensor -> Mat.tensor = "cc_mat_add"

external cc_mat_sub : Mat.tensor -> Mat.tensor -> Mat.tensor = "cc_mat_sub"

(* external cc_mat_sum : Mat.tensor -> float = "cc_mat_sum" *)

external cc_vec_sub : Vec.tensor -> Vec.tensor -> Vec.tensor = "cc_vec_sub"

external cc_vec_add : Vec.tensor -> Vec.tensor -> Vec.tensor = "cc_vec_add"

external cc_vec_sum : Vec.tensor -> float = "cc_vec_sum"

external cc_mat_flatten : Mat.tensor -> Vec.tensor = "cc_mat_flatten"

external cc_mat_flatten_bp : int -> int -> Vec.tensor ->
                             Mat.tensor = "cc_mat_flatten_bp"

external cc_mat3_flatten : Mat3.tensor -> Vec.tensor = "cc_mat3_flatten"

external cc_mat3_flatten_bp : int -> int -> int -> Vec.tensor ->
                             Mat3.tensor = "cc_mat3_flatten_bp"

external gpu_init : unit -> unit = "cc_gpu_init"

external gpu_finish : unit -> unit = "cc_gpu_finish"

external cc_pooling_bp : Mat.tensor -> Mat.tensor -> int -> int -> int -> int ->
                         Mat.tensor =
  "cc_pooling_bp_bytecode" "cc_pooling2d_bp_native"

external cc_conv_bp : Mat.tensor -> Mat.tensor -> Mat.tensor ->
                                 Mat.tensor -> Mat.tensor -> Vec.tensor -> bool -> int ->
                                (Mat.tensor * Mat.tensor * Vec.tensor) =
  "cc_conv_bp_bytecode" "cc_conv2d_bp_native"

external cc_fully_connected_bp : Mat.tensor -> Vec.tensor -> Vec.tensor ->
                                 Vec.tensor -> Mat.tensor -> Vec.tensor -> bool -> int ->
                                (Vec.tensor * Mat.tensor * Vec.tensor) =
  "cc_fully_connected_bp_bytecode" "cc_fully_connected_bp_native"

external cc_pooling3d_ff : Mat3.tensor -> int -> int -> int -> int -> int -> int ->
                           Mat3.tensor =
  "cc_pooling_ff_bytecode" "cc_pooling_ff_native"

let pooling3d_ff (inp : mat3) pool_type stride
      (res_shape : Mat3.shape) (filter_shape : Mat.shape) =
  cc_pooling3d_ff inp.matrix (pooling_to_enum pool_type) stride
    (col res_shape.dim2) (row res_shape.dim1)
    (col filter_shape.dim2) (row filter_shape.dim1)
  |> Mat3.create

external cc_pooling2d_ff : Mat.tensor -> int -> int -> int -> int -> int -> int ->
                           Mat.tensor =
  "cc_pooling_ff_bytecode" "cc_pooling_ff_native"

let pooling2d_ff (inp : mat) pool_type stride
      (res_shape : Mat.shape) (filter_shape : Mat.shape) =
  cc_pooling2d_ff inp.matrix (pooling_to_enum pool_type) stride
    (col res_shape.dim2) (row res_shape.dim1)
    (col filter_shape.dim2) (row filter_shape.dim1)
  |> Mat.create

external cc_conv3d_ff : Mat3.tensor -> Mat3.tensor -> Vec.tensor ->
                      int -> int -> int -> int -> int -> Mat3.tensor =
  "cc_conv_ff_bytecode" "cc_conv_ff_native"

let conv3d_ff (inp : mat3) (kerns : mat3) (bias : vec) actf padding stride (Col resw) (Row resh) =
  cc_conv3d_ff inp.matrix kerns.matrix bias.matrix (actf_to_enum actf) padding stride resw resh
  |> Mat3.create

external cc_conv2d_ff : Mat.tensor -> Mat.tensor -> Vec.tensor ->
                      int -> int -> int -> int -> int -> Mat.tensor =
  "cc_conv_ff_bytecode" "cc_conv_ff_native"

let conv2d_ff (inp : mat) (kerns : mat) (bias : vec) actf padding stride (Col resw) (Row resh) =
  cc_conv2d_ff inp.matrix kerns.matrix bias.matrix (actf_to_enum actf) padding stride resw resh
  |> Mat.create

external cc_mat_scale : float -> Mat.tensor -> Mat.tensor = "cc_mat_scale"

external cc_vec_scale : (float [@unboxed]) -> Vec.tensor -> Vec.tensor = "cc_vec_scale_byte" "cc_vec_scale"

*)
