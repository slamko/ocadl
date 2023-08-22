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

external cc_mat3_random : int -> int -> int -> Mat3.tensor = "cc_mat3_random"

external cc_mat_random : int -> int -> Mat.tensor = "cc_mat_random"

external cc_vec_random : int -> Vec.tensor = "cc_vec_random"

external cc_mat3_nil : int -> int -> int -> Mat3.tensor = "cc_mat3_nil"

external cc_mat_nil : int -> int -> Mat.tensor = "cc_mat_nil"

external cc_vec_nil : int -> Vec.tensor = "cc_vec_nil"

external cc_fully_connected_ff : Vec.tensor -> Mat.tensor -> Vec.tensor ->
  Vec.tensor = "cc_fully_connected_ff"

(* external cc_mat_sub : Mat.tensor -> Mat.tensor -> Mat.tensor = "cc_mat_sub" *)

(* external cc_mat_sum : Mat.tensor -> float = "cc_mat_sum" *)

external cc_vec_sub : Vec.tensor -> Vec.tensor -> Vec.tensor = "cc_vec_sub"

external cc_vec_sum : Vec.tensor -> float = "cc_vec_sum"

external cc_mat_flatten : Mat.tensor -> Vec.tensor = "cc_mat_flatten"

external cc_mat_flatten_bp : int -> int -> Vec.tensor ->
                             Mat.tensor = "cc_mat_flatten_bp"

external gpu_init : unit -> int = "cc_gpu_init"

external cc_fully_connected_bp : Mat.tensor -> Vec.tensor -> Vec.tensor ->
                                 Vec.tensor ->
                                (Vec.tensor * Mat.tensor * Vec.tensor) =
  "cc_fully_connected_bp"


external cc_mat_scale : float -> Mat.tensor -> Mat.tensor = "cc_mat_scale"

external cc_vec_scale : float -> Vec.tensor -> Vec.tensor = "cc_vec_scale"
