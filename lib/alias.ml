open Bigarray

type activation = float -> float [@@deriving show]

type deriv = float -> float [@@deriving show]

type mat3 = (float, float32_elt, c_layout) Array3.t 
type mat  = (float, float32_elt, c_layout) Array2.t 
type vec  = (float, float32_elt, c_layout) Array1.t 

module New = struct
  type t = {b : float}
end

module Bew = struct
  type t = {a : int}
end


module Some = struct
type c = New.t
type x = Bew.t
end

open Some

type _ tensor =
  | Tensor1 : vec -> vec tensor
  | Tensor2 : mat -> mat tensor
  | Tensor3 : mat3 -> mat3 tensor

(* let matr : type a. a tensor -> a tensor -> unit = *)
  (* fun a b -> *)
  (* match a, b with *)
  (* | Tensor1 _, Tensor3 _ -> () *)

let make_tens1 v = Tensor1 v
let make_tens2 v = Tensor2 v
let make_tens3 v = Tensor3 v
