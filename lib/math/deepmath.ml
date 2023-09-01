open Common
open Types
open Alias
open Bigarray
open Tensor

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
