open Common
open Types
open Matrix

type _ shape =
  | ShapeMatVec : (MatShape.shape * VectorShape.shape)  -> mat vector shape
  | ShapeMatMat : (MatShape.shape * MatShape.shape)     -> mat matrix shape
  | ShapeVec    : (VectorShape.shape) -> vec shape
  | ShapeMat    : (MatShape.shape)    -> mat shape

let shape_size shape =
  match shape with
  | ShapeMatVec (m, v) -> MatShape.shape_size m * VectorShape.shape_size v
  | ShapeMatMat (m1, m2) -> MatShape.shape_size m1 * MatShape.shape_size m2
  | ShapeVec v -> VectorShape.shape_size v
  | ShapeMat v -> MatShape.shape_size v

let get_shape : type a. a -> a shape =
  fun tensor ->


let get_shape mat =
  
  make_shape mat.rows mat.cols

let shape_eq shape1 shape2 =
  if compare shape1 shape2 <> 0
  then false
  else true

let empty_shape () =
  { dim1 = Row 0;
    dim2 = Col 0;
    dim3 = 1;
  }

let shape_match mat1 mat2 =
  let shape = get_shape mat2 in
  match get_shape mat1 |> compare shape with
  | 0 -> ()
  | _ -> failwith "Matrix shapes do not match."
 
