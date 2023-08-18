open Common
open Matrix
open Alias

type _ shape =
  | ShapeMatVec : (Mat.shape * Vec.shape) -> mat vector tensor shape
  | ShapeMat    : (Mat.shape) -> mat tensor shape
  | ShapeVec    : (Vec.shape) -> vec tensor shape
  | ShapeMatMat : (Mat.shape * Mat.shape) -> mat matrix tensor shape

let shape_size : type a. a shape -> int =
  function 
  | ShapeMatVec (m, v) -> Mat.shape_size m * Vec.shape_size v
  | ShapeMatMat (m1, m2) -> Mat.shape_size m1 * Mat.shape_size m2
  | ShapeVec v -> Vec.shape_size v
  | ShapeMat v -> Mat.shape_size v

let get_shape : type a. a tensor -> a tensor shape =
  fun tensor ->
  match tensor with
  | Tensor1 x -> ShapeVec (Vec.get_shape x)
  | Tensor2 x -> ShapeMat (Mat.get_shape x)
  | Tensor3 vec ->
     (match Vec.get_first_opt vec with
     | Some first -> ShapeMatVec (Mat.get_shape first, Vec.get_shape vec)
     | None -> ShapeMatVec (Mat.shape_zero (), Vec.get_shape vec)
     )
  | Tensor4 mat ->
     (match Mat.get_first_opt mat with
      | Some first -> ShapeMatMat (Mat.get_shape first, Mat.get_shape mat)
      | None -> ShapeMatMat (Mat.shape_zero (), Mat.get_shape mat)
     )

let shape_eq : type a. a shape -> a shape -> bool =
  fun shape1 shape2 ->
  match shape1, shape2 with
  | ShapeMatVec (m1, v1), ShapeMatVec (m2, v2) ->
     (if compare m1 m2 <> 0 || compare v1 v2 <> 0
      then false
      else true)
  | ShapeMatMat (m1, m3), ShapeMatMat (m2, m4) ->
     (if compare m1 m2 <> 0 || compare m3 m4 <> 0
      then false
      else true)
  | ShapeVec v1, ShapeVec v2 ->
     (if compare v1 v2 <> 0 
      then false
      else true)
  | ShapeMat m1, ShapeMat m2 ->
     (if compare m1 m2 <> 0 
      then false
      else true) 

let make_shape_vec vec_shape =
  ShapeVec vec_shape

let make_shape_mat mat_shape =
  ShapeMat mat_shape

let make_shape_mat_mat mat1_shape  mat2_shape =
  ShapeMatMat (mat1_shape, mat2_shape)

let make_shape_mat_vec mat1_shape vec2_shape =
  ShapeMatVec (mat1_shape, vec2_shape)

let shape_match mat1 mat2 =
  let shape = get_shape mat2 in
  match get_shape mat1 |> compare shape with
  | 0 -> ()
  | _ -> failwith "Matrix shapes do not match."
 
