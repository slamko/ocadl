module Mat = Matrix.Mat
module Vec = Matrix.Vector

type activation = float -> float [@@deriving show]

type deriv = float -> float [@@deriving show]

type mat = float Mat.t [@@deriving show]

type vec = float Vec.t [@@deriving show]

type 'a matrix = 'a Mat.t [@@deriving show]
type 'a vector = 'a Vec.t [@@deriving show]

type _ tensor =
  | Tensor1 : vec -> vec tensor
  | Tensor2 : mat -> mat tensor
  | Tensor3 : mat vector -> mat vector tensor
  | Tensor4 : mat matrix -> mat matrix tensor

let make_tens1 v = Tensor1 v
let make_tens2 v = Tensor2 v
let make_tens3 v = Tensor3 v
let make_tens4 v = Tensor4 v
