open Common
open Deepmath

type shape_mat1 = {
    dim1 : row;
    dim2 : col;
  } [@@deriving show]

type shape_mat2 = {
    dim1 : row;
    dim2 : col;
    dim3 : row;
    dim4 : col;
  } [@@deriving show]

type _ shape =
  | ShapeMat1 : vec matrix -> shape_mat1 shape
  | ShapeMat2

let shape_size shape =
  get_row shape.dim1
  |> ( * ) @@ get_col shape.dim2 
  |> ( * ) shape.dim3


let make_shape3d dim1 dim2 dim3 = 
  if get_row dim1 < 0 || get_col dim2 < 0 || dim3 < 0
  then failwith "Invalid shape."
  else { dim1;
         dim2;
         dim3;
       }

let make_shape rows cols = make_shape3d rows cols 1

let get_shape mat =
  
  make_shape mat.rows mat.cols

let shape_eq shape1 shape2 =
  if compare shape1 shape2 <> 0
  then false
  else true

let of_shape init_val shape =
  make shape.dim1 shape.dim2 init_val

let empty_shape () =
  { dim1 = Row 0;
    dim2 = Col 0;
    dim3 = 1;
  }

let zero_of_shape shape =
  of_shape 0. shape

let random_of_shape shape =
  random shape.dim1 shape.dim2

let reshape_of_shape mat shape  =
  reshape shape.dim1 shape.dim2 mat

let shape_match mat1 mat2 =
  let shape = get_shape mat2 in
  match get_shape mat1 |> compare shape with
  | 0 -> ()
  | _ -> failwith "Matrix shapes do not match."
 
