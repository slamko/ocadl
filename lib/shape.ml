open Common

type shape = {
    dim1 : row;
    dim2 : col;
    dim3 : int;
  } [@@deriving show]

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

let shape_match mat1 mat2 =
  let shape = get_shape mat2 in
  match get_shape mat1 |> compare shape with
  | 0 -> ()
  | _ -> failwith "Matrix shapes do not match."
 
