module Mat = Matrix.Mat
module Vec = Matrix.Vector

type mat = float Mat.t [@@deriving show]

type vec = float Vec.t [@@deriving show]

type 'a matrix = 'a Mat.t [@@deriving show]
type 'a vector = 'a Vec.t [@@deriving show]

