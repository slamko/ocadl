open Bigarray
open Tensor

type activation = float -> float [@@deriving show]

type deriv = float -> float [@@deriving show]

type mat3 = Mat3.t
type mat  = Mat.t
type vec  = Vec.t

type actf =
  | Sigmoid [@value 0]
  | Relu
[@@deriving enum]

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

let make_tens1 v = Tensor1 v
let make_tens2 v = Tensor2 v
let make_tens3 v = Tensor3 v
