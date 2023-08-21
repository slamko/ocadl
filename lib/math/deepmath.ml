open Common
open Types
open Alias

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

